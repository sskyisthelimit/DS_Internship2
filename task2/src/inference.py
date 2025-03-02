import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

from nlp.inference import (
    predict_ner_labels, post_process_tokens_and_labels,
    model, tokenizer)

from cv.inference import predict_animal, unique_labels

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Example of parsing multiple named arguments.')
    parser.add_argument('--image_path', type=str,
                        help='Path for image file.', required=True)
    parser.add_argument('--sentence', type=str,
                        help='A sentence string.', required=True)
    parser.add_argument('--device', type=str,
                        help='A device string.', required=True)

    args = parser.parse_args()
    
    nlp_model = model.to(args.device)

    pred_tokens, pred_labels = predict_ner_labels(
        args.sentence,
        model=nlp_model,
        tokenizer=tokenizer,
        device=args.device,
        print_pairs=False)
    
    RELEASE_ASSET_URL = "https://github.com/sskyisthelimit/DS_Internship2/releases/download/v1.0.0/animals_checkpoint.pth"
    cv_weights_dir = "../../weights"
    os.makedirs(cv_weights_dir, exist_ok=True)
    cv_weights_path = os.path.join(cv_weights_dir, "animals_checkpoint.pth")
    from cv.utils import download_file
    download_file(RELEASE_ASSET_URL, cv_weights_path)
    clsf_pred = predict_animal(img_path=args.image_path,
                               weights_path=cv_weights_path)
    
    clsf_pred_label = clsf_pred.tolist()[0]
    clsf_pred_name = unique_labels[clsf_pred_label] 

    processed_tokens, processed_labels = post_process_tokens_and_labels(
        pred_tokens, pred_labels)
    
    probable_class_names = []

    for token, label in zip(processed_tokens, processed_labels):
        if label != 0:
            probable_class_names.append(token)

    is_true_animal = clsf_pred_name in probable_class_names

    print(is_true_animal, f"predicted name of animal: {clsf_pred_name}", )
    


        