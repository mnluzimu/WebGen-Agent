import os
import re
import html
from pathlib import Path


vite_file_content = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 0, // Will use random available port
    strictPort: false,
    watch: {
      usePolling: true,
      interval: 1000
    },
    hmr: {
      port: 0 // Random port for HMR
    }
  },
  preview: {
    port: 0 // Random port for preview
  }
})"""


def extract_and_write_files(response: str, workspace_dir: str):
    """
    Parses a string with <boltAction type="file" filePath="...">...</boltAction> blocks
    and writes the files to the corresponding paths under workspace_dir.
    """
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Find all boltAction blocks
    pattern = r'<boltAction type="file" filePath="(.*?)">(.*?)</boltAction>'
    matches = re.findall(pattern, response, flags=re.DOTALL)
    
    for file_path, file_content in matches:
        # Decode HTML entities (e.g., &lt; becomes <)
        decoded_content = html.unescape(file_content)
        
        # Create full file path
        full_path = os.path.join(workspace_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write the file content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(decoded_content)
        print(f"Created: {full_path}")

    # Create full file path
    full_path = os.path.join(workspace_dir, "vite.config.js")
    if not os.path.isfile(full_path):
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write the file content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(vite_file_content)
        print(f"Created: {full_path}")


def get_sorted_file_paths(workspace_root):
    """
    Returns a list of relative file paths under workspace_root.
    Files in subdirectories are listed before files in their parent directory.
    Entries are ordered lexicographically.
    """
    workspace_root = Path(workspace_root).resolve()
    all_files = []

    for dirpath, dirnames, filenames in os.walk(workspace_root):
        dirnames[:] = [d for d in dirnames if d != "node_modules"]

        # Sort directories and filenames to ensure consistent ordering
        dirnames.sort()
        filenames.sort()

        rel_dir = Path(dirpath).relative_to(workspace_root)

        # Add files from subdirectories first due to os.walk's top-down order
        for filename in filenames:
            file_path = (rel_dir / filename).as_posix()
            all_files.append(file_path)

    # Custom sort: sort by depth (more nested first), then lexicographically
    def sort_key(path):
        parts = path.split('/')
        return (len(parts), parts)

    return sorted(all_files, key=sort_key, reverse=True)