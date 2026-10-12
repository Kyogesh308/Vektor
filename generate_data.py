import random
import os

subjects = ['The spaceship', 'The engineer', 'A database', 'The AI', 'A cat', 'The system', 'An astronaut']
verbs = ['optimized', 'queried', 'explored', 'navigated', 'analyzed', 'computed', 'observed']
objects = ['the galaxy.', 'the vector space.', 'the latency metrics.', 'a distant planet.', 'the data structures.', 'a small mouse.', 'the anomaly.']

output_path = os.path.join(os.path.dirname(__file__), 'large_dummy_corpus.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for i in range(5000):
        # We add some length so the text splitter produces separate chunks
        text = f"Document segment {i}: {random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)} "
        text += f"This provides extra context for segment {i} to ensure the text chunker breaks it up. " * 3
        f.write(text + '\n')

print(f"Generated {output_path}")
