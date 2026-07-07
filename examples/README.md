# Example Test Clips

This folder is a place to drop short football clips for end-to-end testing.

Good sources to check:

- Internet Archive: search for "football match" or "soccer game"
- Wikimedia Commons: some classic football clips are available there
- Open sports datasets on Kaggle or Hugging Face

For best results, use a 2-10 minute clip that contains a mix of events such as a goal, card, save, shot, or substitution.

Typical workflow:

```bash
# 1. Drop a clip here, for example examples/test_clip.mp4
# 2. Run the generator
python -m football_highlights examples/test_clip.mp4 --backend ollama

# 3. Inspect the generated files
ls -lh output/
```

The `output/` directory is gitignored, so generated highlights and reports stay local.
