#!/usr/bin/env bash

echo "🚀 Deploying snapshot to Hugging Face..."

# Push a clean tree without history to avoid binary file rejections
git push huggingface $(git commit-tree HEAD^{tree} -m "Automated Deploy to HF"):main --force

echo "✅ Deployment complete!"