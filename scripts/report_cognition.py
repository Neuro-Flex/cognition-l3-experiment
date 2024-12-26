import sys
import os
import torch
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.consciousness_model import ConsciousnessModel

def plot_cognition_progress(history):
    """Plot the cognition progress over time."""
    epochs = list(range(1, len(history) + 1))
    
    # Handle both old and new format
    progress = []
    for entry in history:
        if isinstance(entry, dict):
            # Try both potential keys
            if 'cognition_progress' in entry:
                progress.append(entry['cognition_progress'])
            elif 'total' in entry:
                progress.append(entry['total'])
            else:
                raise KeyError("Entry missing both 'cognition_progress' and 'total' keys")
        else:
            progress.append(float(entry))  # Handle direct numerical values

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, progress, marker='o', linestyle='-', color='b')
    plt.title('Cognition Progress Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Cognition Progress (%)')
    plt.grid(True)
    plt.savefig('cognition_progress.png')
    plt.show()

def main():
    # Initialize the model
    config = ConsciousnessModel.create_default_config()
    model = ConsciousnessModel(**config)
    
    # Define the metrics
    metrics = {
        'phi': 0.85,
        'coherence': 0.85,
        'stability': 0.85,
        'adaptability': 0.85,
        'memory_retention': 0.85,
        'emotional_coherence': 0.85,
        'decision_making_efficiency': 0.85
    }
    
    # Calculate cognition progress
    model.calculate_cognition_progress(metrics)
    
    # Report cognition progress
    report = model.report_cognition_progress()
    print(report)

    # Plot cognition progress
    plot_cognition_progress(model.cognition_progress_history)

if __name__ == "__main__":
    main()
