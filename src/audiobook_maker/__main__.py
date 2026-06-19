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
@click.option("--voice-map", type=click.Path(exists=True), help="Voice configuration JSON")
def synthesize(project_dir, voice_map):
    """Synthesize annotated script into audio."""
    click.echo(f"Synthesizing {project_dir}...")
    click.echo("(Not yet implemented)")


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
def extract_voices(audiobook_path, output):
    """Extract character voices from an existing audiobook via diarization."""
    click.echo(f"Extracting voices from {audiobook_path}...")
    click.echo("(Not yet implemented)")


if __name__ == "__main__":
    cli()
