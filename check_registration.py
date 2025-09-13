#!/usr/bin/env python3
# Quick test of technique registration
from raglib.techniques.fixed_size_chunker import FixedSizeChunker
from raglib.techniques.sentence_window_chunker import SentenceWindowChunker
from raglib.techniques.semantic_chunker import SemanticChunker
from raglib.registry import TechniqueRegistry

print('New chunking techniques registered:')
for name, cls in TechniqueRegistry.list().items():
    if 'chunk' in name.lower():
        print(f'  - {name}: {cls.__name__}')

required = ['fixed_size_chunker', 'sentence_window_chunker', 'semantic_chunker']
all_registered = all(name in TechniqueRegistry.list() for name in required)
print(f'\nAll required techniques registered: {all_registered}')
