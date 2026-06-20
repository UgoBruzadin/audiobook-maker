"""CLI entry point for audiobook-maker."""

import click


@click.group()
def cli():
    """Multi-voice audiobook generator from EPUBs."""
    pass


@cli.command()
@click.argument("epub_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="./project", help="Output project directory")
def parse(epub_path, output):
    """Parse an EPUB into structured chapters."""
    from .parse import parse_epub
    import json
    from pathlib import Path

    book = parse_epub(epub_path)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save parsed book as JSON
    data = {
        "title": book.title,
        "author": book.author,
        "chapters": [
            {
                "index": ch.index,
                "title": ch.title,
                "paragraphs": [
                    {
                        "text": p.text,
                        "is_dialogue": p.is_dialogue,
                        "is_emphasis": p.is_emphasis,
                        "is_blockquote": p.is_blockquote,
                    }
                    for p in ch.paragraphs
                ],
            }
            for ch in book.chapters
        ],
    }

    out_path = out_dir / "parsed_book.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    click.echo(f"Parsed: {book.title} by {book.author}")
    click.echo(f"  {len(book.chapters)} chapters, {book.word_count:,} words")
    click.echo(f"  Saved to: {out_path}")


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.option("--llm-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible LLM API base URL")
@click.option("--llm-model", default="qwen3-30b", help="Model name for annotation")
@click.option("--llm-api-key", default="not-needed", help="API key (if required by LLM server)")
@click.option("--review/--no-review", default=True, help="Run review pass to fix attribution errors")
def annotate(project_dir, llm_base_url, llm_model, llm_api_key, review):
    """Annotate parsed book with speaker attribution."""
    import json
    from pathlib import Path
    from .annotate import annotate_book, review_script, save_script, AnnotationConfig

    project = Path(project_dir)
    parsed_path = project / "parsed_book.json"

    if not parsed_path.exists():
        click.echo(f"Error: {parsed_path} not found. Run 'parse' first.", err=True)
        raise SystemExit(1)

    with open(parsed_path) as f:
        parsed_book = json.load(f)

    click.echo(f"Annotating '{parsed_book.get('title', '?')}' with {llm_model}...")

    config = AnnotationConfig(
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
    )

    entries = annotate_book(parsed_book, config)

    if review:
        click.echo("Running review pass...")
        entries = review_script(entries, config)

    save_script(entries, project / "annotated_script.json")
    click.echo(f"Done! {len(entries)} script entries saved.")


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.option("--voice-map", "-v", type=click.Path(exists=True), required=True, help="Voice configuration JSON")
@click.option("--engine", type=click.Choice(["xtts_v2", "qwen3_tts"]), default="xtts_v2", help="TTS engine")
@click.option("--language", "-l", default="en", help="Language code")
@click.option("--speed", type=float, default=1.0, help="Playback speed")
@click.option("--max-chunk", type=int, default=350, help="Max characters per TTS chunk")
def synthesize(project_dir, voice_map, engine, language, speed, max_chunk):
    """Synthesize annotated script into audio."""
    from pathlib import Path
    from .annotate import load_script
    from .synthesize import synthesize_script, load_voice_map, SynthesisConfig

    project = Path(project_dir)
    script_path = project / "annotated_script.json"

    if not script_path.exists():
        click.echo(f"Error: {script_path} not found. Run 'annotate' first.", err=True)
        raise SystemExit(1)

    script = load_script(script_path)
    voices = load_voice_map(voice_map)

    click.echo(f"Synthesizing {len(script)} entries with {engine}...")
    click.echo(f"  Voices: {', '.join(voices.keys())}")

    config = SynthesisConfig(
        engine_name=engine,
        max_chunk_chars=max_chunk,
        language=language,
        speed=speed,
    )

    rendered = synthesize_script(
        script=script,
        voice_map=voices,
        output_dir=project / "audio",
        config=config,
    )

    click.echo(f"Done! {len(rendered)} audio clips rendered.")


@cli.command()
@click.argument("project_dir", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["m4b", "mp3", "wav"]), default="m4b")
@click.option("--output", "-o", help="Output file path")
def export(project_dir, fmt, output):
    """Export rendered audio as M4B/MP3."""
    click.echo(f"Exporting {project_dir} as {fmt}...")
    click.echo("(Not yet implemented)")


@cli.command()
@click.argument("audiobook_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="./voices", help="Output directory for extracted voice clips")
@click.option("--num-speakers", type=int, default=None, help="Exact number of speakers (if known)")
@click.option("--min-speakers", type=int, default=None, help="Minimum expected speakers")
@click.option("--max-speakers", type=int, default=None, help="Maximum expected speakers")
@click.option("--hf-token", default=None, help="HuggingFace token for pyannote models")
@click.option("--min-clip", type=float, default=5.0, help="Minimum clip duration (seconds)")
@click.option("--target-clip", type=float, default=20.0, help="Target clip duration for cloning")
def extract_voices(audiobook_path, output, num_speakers, min_speakers, max_speakers, hf_token, min_clip, target_clip):
    """Extract character voices from an existing audiobook via diarization."""
    from pathlib import Path
    from .voices.extract import diarize, compute_speaker_profiles, extract_clips

    click.echo(f"Extracting voices from {audiobook_path}...")

    # Step 1: Diarize
    result = diarize(
        audiobook_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
    )

    # Step 2: Compute speaker profiles (embeddings + best clips)
    click.echo("Computing speaker profiles...")
    clustering = compute_speaker_profiles(result)

    # Step 3: Extract and save clips
    click.echo("Extracting reference clips...")
    voices = extract_clips(
        clustering,
        output_dir=output,
        target_duration=target_clip,
        min_clip_duration=min_clip,
    )

    click.echo(f"\nDone! Extracted {len(voices)} clips for {len(clustering.profiles)} speakers → {output}")


if __name__ == "__main__":
    cli()
