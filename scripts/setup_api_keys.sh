#!/bin/bash
# Script to set up API keys for the new models

# Check if API keys are already set
if [ -z "$GOOGLE_API_KEY" ]; then
  echo "Please enter your Google API key for Gemini models:"
  read -s GOOGLE_API_KEY
  echo "export GOOGLE_API_KEY=$GOOGLE_API_KEY" >> ~/.bashrc
  echo "export GOOGLE_API_KEY=$GOOGLE_API_KEY" >> ~/.zshrc
  echo "Google API key has been added to your shell configuration files."
else
  echo "Google API key is already set."
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Please enter your Anthropic API key for Claude models:"
  read -s ANTHROPIC_API_KEY
  echo "export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> ~/.bashrc
  echo "export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> ~/.zshrc
  echo "Anthropic API key has been added to your shell configuration files."
else
  echo "Anthropic API key is already set."
fi

# Remind about OpenAI API key if not set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Please enter your OpenAI API key for GPT models:"
  read -s OPENAI_API_KEY
  echo "export OPENAI_API_KEY=$OPENAI_API_KEY" >> ~/.bashrc
  echo "export OPENAI_API_KEY=$OPENAI_API_KEY" >> ~/.zshrc
  echo "OpenAI API key has been added to your shell configuration files."
else
  echo "OpenAI API key is already set."
fi

echo ""
echo "API keys have been configured. Please restart your terminal or run 'source ~/.bashrc' (or 'source ~/.zshrc' if using zsh) to apply the changes."
echo "You can now use the new models: GEMINI_2_PRO, CLAUDE_3_7_SONNET, O3_MINI, and GPT_4O."