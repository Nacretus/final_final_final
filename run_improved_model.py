import os
import torch
import numpy as np
import pickle
from simple_model_architecture import SimplifiedCharCNNBiLSTM
from classifier_chain_model import ClassifierChainModel, MCDropoutChainModel
from classifier_chain_integration import batch_predict_with_chain, interactive_chain_prediction, batch_predict_with_uncertainty
from OOD_evaluation_function import evaluate_ood_performance
from CONFIG import CONFIG

def load_model_and_vocab():
    """
    Load the saved model and vocabulary.
    """
    model_path = os.path.join(CONFIG['output_dir'], 'chain_model.pth')
    vocab_path = os.path.join(CONFIG['output_dir'], 'char_vocab.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print(f"Error: Model ({model_path}) or vocabulary ({vocab_path}) not found.")
        print("Please train the model first.")
        return None, None
    
    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        char_vocab = pickle.load(f)
    
    # Create the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_model = SimplifiedCharCNNBiLSTM(
        n_chars=char_vocab.n_chars,
        char_emb_dim=CONFIG['char_emb_dim'],
        lstm_hidden_dim=CONFIG['lstm_hidden_dim'],
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)
    
    chain_model = ClassifierChainModel(base_model).to(device)
    
    # Load model weights
    print(f"Loading model from {model_path}...")
    chain_model.load_state_dict(torch.load(model_path, map_location=device))
    chain_model.eval()
    
    print("Model and vocabulary loaded successfully.")
    return chain_model, char_vocab

def test_examples(model, char_vocab):
    """
    Test model on a few examples that were problematic before.
    """
    examples = [
        "goodafternoon this is a not toxic comment",
        "i love you",
        "things need to adjust",
        "nakita mo ba yung comment in sir kanina sa gc",
        "bitch isa pa pasasagasa kita sa truck",
        "you fucking suck",
        "I really like your work",
        "kill yourself",
        "this is a normal comment, hello"
    ]
    
    print("\n=== Testing Improved Model on Example Phrases ===")
    
    try:
        # Get predictions with standard method first
        print("\nStandard Predictions:")
        standard_results = batch_predict_with_chain(model, examples, char_vocab)
        
        # Get predictions with MC dropout if possible
        print("\nMonte Carlo Dropout Predictions:")
        try:
            mc_results = batch_predict_with_uncertainty(model, examples, char_vocab, num_samples=20)
            mc_available = True
        except Exception as e:
            print(f"Error with MC dropout: {e}")
            print("Using standard results instead")
            mc_results = standard_results
            mc_available = False
        
        # Print results side by side
        print("\n" + "="*80)
        print(f"{'Text':<40} {'Standard':<15} {'MC Dropout':<15} {'MC Uncertainty'}")
        print("="*80)
        
        for i, example in enumerate(examples):
            std_toxicity = standard_results[i]['toxicity']['label']
            std_level = standard_results[i]['toxicity']['level']
            
            if mc_available:
                mc_toxicity = mc_results[i]['toxicity']['label']
                mc_level = mc_results[i]['toxicity']['level']
                
                if 'uncertainty' in mc_results[i]:
                    uncertainty = mc_results[i]['uncertainty']['overall']
                    uncertainty_str = f"{uncertainty:.4f}"
                else:
                    uncertainty_str = "N/A"
            else:
                mc_toxicity = "N/A"
                mc_level = "N/A"
                uncertainty_str = "N/A"
            
            # Display truncated text if too long
            display_text = example[:37] + '...' if len(example) > 37 else example
            
            print(f"{display_text:<40} "
                  f"{std_toxicity} ({std_level})  "
                  f"{mc_toxicity} ({mc_level if isinstance(mc_level, (int, float)) else mc_level})     "
                  f"{uncertainty_str}")
            
            # Print categories if toxic
            if std_level > 0:
                print("  Categories:")
                # Standard categories
                std_cats = [cat for cat, info in standard_results[i]['categories'].items() if info['detected']]
                
                if mc_available and mc_level not in ['N/A'] and mc_level > 0:
                    mc_cats = [cat for cat, info in mc_results[i]['categories'].items() if info['detected']]
                    print(f"  Standard: {', '.join(std_cats) if std_cats else 'None'}")
                    print(f"  MC: {', '.join(mc_cats) if mc_cats else 'None'}")
                else:
                    print(f"  Standard: {', '.join(std_cats) if std_cats else 'None'}")
            print("-"*80)
    
    except Exception as e:
        print(f"Error testing examples: {e}")
        import traceback
        traceback.print_exc()

def run_ood_evaluation(model, char_vocab):
    """
    Run OOD evaluation on the model.
    """
    ood_data_path = os.path.join(CONFIG['output_dir'], 'ood_test_data.csv')
    
    if not os.path.exists(ood_data_path):
        print(f"OOD data file not found at {ood_data_path}")
        print("Creating OOD dataset...")
        from dataprocessing import create_ood_test_set
        create_ood_test_set(CONFIG['data_path'], ood_data_path, criteria='long_texts')
    
    results = evaluate_ood_performance(model, char_vocab, CONFIG['data_path'], ood_data_path)
    return results

def main():
    """
    Main function to run the improved model.
    """
    print("=== Running Improved Toxicity Detection Model ===")
    
    # Load model and vocabulary
    model, char_vocab = load_model_and_vocab()
    if model is None or char_vocab is None:
        return
    
    while True:
        print("\nSelect an option:")
        print("1. Test on example phrases")
        print("2. Run OOD evaluation")
        print("3. Start interactive prediction")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            test_examples(model, char_vocab)
        elif choice == '2':
            run_ood_evaluation(model, char_vocab)
        elif choice == '3':
            interactive_chain_prediction(model, char_vocab, use_mc_dropout=True)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()