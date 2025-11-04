import json
import os
import argparse
import re

def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"Warning: {file_path} not found.")
        return {}
    
def clean_fallacy_data(data):
    """
    Clean up fallacy data in the merged JSON, replacing 'No fallacies found' with 'None'
    and normalizing fallacy names like 'strawman' and 'straw man' to 'strawman fallacy'.
    """
    # fallacy_normalizations = {}
    fallacy_normalizations = {
        "straw_man_fallacy" : "strawman fallacy",
        "straw man fallacy" : "strawman fallacy",
        "strawman": "strawman fallacy",
        "straw man": "strawman fallacy",
        "straw_man": "strawman fallacy",
        "straw_man_argument": "strawman fallacy",
        "no fallacies found": "none",
        "no fallacies": "none",
        "no fallacy": "none",
        "no_fallacies_found": "none",
        "no fallacy found": "none",
        "no fallacy present": "none",
        "no": "none",
        "no_fallacy_found": "none",
        "no_fallacy_detected": "none",
        "no_fallacy_present": "none",
        "ad hom": "ad_hominem",
        "ad_hom": "ad_hominem",
        "false_cause" : "false_cause",
        "false_causality": "false_cause",
        "false causality": "false_cause",
        "false dichotomy": "false_dilemma",
        "false dichotomy": "false_dilemma",
        "false_cause_fallacy": "false_cause",
        "false_dichotomy": "false_dilemma",
        "argument_from_authority" : "appeal_to_authority",
        "argument_from_ignorance" : "appeal_to_ignorance",
        "argument_from_authority" : "appeal_to_authority",
        "overgeneralization" : "hasty_generalization",
        "sweeping_generalization" : "hasty_generalization",
        "generalization" : "hasty_generalization",
        "simplification" : "oversimplification",
        "circular_argument" : "circular_reasoning",
        "begging_the_question": "circular_reasoning",



    }

    for question, responses in data.items():
        for response_obj in responses:
            for det_name, det_out in response_obj["fallacy_detectors"].items():
                if "fallacies" in det_out.keys():
                    cur_falls = det_out["fallacies"]
                    # Normalize the fallacy names
                    for i, fallacy in enumerate(cur_falls):
                        normalized_fallacy = fallacy.lower().replace(" ", "_")
                        if normalized_fallacy in fallacy_normalizations:
                            cur_falls[i] = fallacy_normalizations[normalized_fallacy]
    return data


def combine_fallacy_detections(input_file, output_file, folder_mapping, stats_file):
    """
    Combine fallacy detections from multiple models into one structured file.

    input_file: JSON file with model responses (e.g., results_speech_phi4-mini_4bit_all_short.json).
    output_file: Path to save the merged fallacy detection results.
    folder_mapping: Dictionary mapping detector names to their respective folder paths.
    """

    # Path to the main response file
    input_path = f"results/speech_results/results_speech_{input_file}"
    data = load_json(input_path)

    # Load detector files
    detector_data = {}
    for detector, folder in folder_mapping.items():
        detector_file = f"{folder}/analysed_{input_file}"
        detector_data[detector] = load_json(detector_file)
        print("current detector_file processed:", detector_file)

    # Merge fallacy detections
    for question, responses in data.items():
        for response_obj in responses:
            # if "fallacy_detectors" not in response_obj:
            response_obj["fallacy_detectors"] = {} #add the fallacy_detector attribute for each response in input file

            for detector_name, det_data in detector_data.items():
                if question in det_data:
                    current_detector_analysis = det_data[question]
                    response_obj["fallacy_detectors"].update(current_detector_analysis[0]["fallacy_detectors"])
                    #     # if det_response["speaker"] == response_obj["speaker"]:
                    #     response_obj["fallacy_detectors"][detector_name] = det_response["fallacy_detectors"].get(detector_name, {})

    data = clean_fallacy_data(data)

    # Save the merged output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure output folder exists
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Combined fallacy analysis saved to {output_file}")


###################################
"""
statistics about given input files.
json structure: (file called args.stats_name.json)
- input_file: input_file_name {
    different fallacies found: [list of different fallacies (names)],
    #questions where 1 fallacy is found in common between all 3,
    #questions where 0 fallacy is found in common between all 3,
    #questions where ALL fallacies are is found between all 3,
    #questions where 1 common fallacy is found between all 3,
    #average number of fallacies found for "FallacyModel/checkpoint_fallacious_phi4mini2", "FallacyModel/checkpoint_fallacious_mistral", "qwen_normal"

}
"""
def calculate_fallacy_statistics(data, detector_names):
    """
    Calculate statistics about fallacies across multiple detectors.

    data: Combined data from detectors.
    detector_names: List of detector names to track.
    """
    cur_index = 0
    all_different_fallacy_types_found = set()
    num_ques_some_fallacy_common_to_all = 0 # number of questions where all fallacies found by all models are common
    num_ques_0_fall_common_to_any = 0 # number of questions where 0 fallcies are shared by any pair of models
    num_questions = 0 #just the total number of questions
    num_questions_found_fallacy_phi = 0 #number of questions where phi found at least 1 fallacy
    num_questions_found_fallacy_mistral = 0 #number of questions where mistral found at least 1 fallacy
    num_questions_found_fallacy_qwen = 0 #number of questions where qwen found at least 1 fallacy
    num_questions_found_fallacy_mistral24 = 0
    num_fallacies_total_phi = 0 #total number of fallacies found by phi, across all questions
    num_fallacies_total_mistral = 0 #total number of fallacies found by mistral, across all questions
    num_fallacies_total_qwen = 0 #total number of fallacies found by qwen, across all questions
    num_fallacies_total_mistral24 = 0
    num_questions_with_zero_fallacies = 0 
    num_agree_phi_mistral = 0
    num_agree_phi_qwen = 0
    num_agree_mistral_qwen = 0
    num_agree_mistral24_qwen = 0
    num_agree_mistral24_phi = 0
    num_agree_mistral24_mistral = 0

    fallacy_frequency = {}

    # Loop through the questions, and the responses (model resp + fallacy detections by all 4)
    for question, responses in data.items():
        num_questions += 1

        fallacies_in_question = {detector: set() for detector in detector_names}

        # Collect fallacies detected by each model for the current question
        for response_obj in responses:
            current_resp_detections = response_obj.get("fallacy_detectors", {})

            for detector, fallacy_and_exp in current_resp_detections.items():
                if isinstance(fallacy_and_exp, str):  
                    try:
                        fallacy_and_exp = json.loads(fallacy_and_exp)
                        if not isinstance(fallacy_and_exp, dict):
                            fallacy_and_exp = {}  
                    except json.JSONDecodeError:
                        fallacy_and_exp = {}

                # Extract fallacies safely
                fallacies_found_by_current_model_in_current_question = fallacy_and_exp.get('fallacies', []) if isinstance(fallacy_and_exp, dict) else []
                if not isinstance(fallacies_found_by_current_model_in_current_question, list):
                    fallacies_found_by_current_model_in_current_question = [fallacies_found_by_current_model_in_current_question]

                for fall in fallacies_found_by_current_model_in_current_question:
                    normalized_fallacy = fall.lower().replace(" ", "_").strip()
                    all_different_fallacy_types_found.add(normalized_fallacy)
                    if normalized_fallacy != "none": # add to fallacies found unless it no fallacy
                        fallacies_in_question[detector].add(normalized_fallacy)
                        fallacy_frequency[normalized_fallacy] = fallacy_frequency.get(normalized_fallacy, 0) + 1
                if not isinstance(fallacies_found_by_current_model_in_current_question, list):
                    print(f"Unexpected structure in {detector}: {fallacies_found_by_current_model_in_current_question}")

        # Convert the collected fallacies to a dictionary format for easier processing
        list_fallacies_in_question = [(model, list(fallacies)) for model, fallacies in fallacies_in_question.items()]
        dict_falls_in_question = {
            model: set(fallacies) for model, fallacies in list_fallacies_in_question
        }

        if all(len(falls) == 0 for falls in dict_falls_in_question.values()):
            num_questions_with_zero_fallacies += 1

        if len(dict_falls_in_question['FallacyModel/checkpoint_fallacious_phi4mini2'] & dict_falls_in_question['FallacyModel/checkpoint_fallacious_mistral']) > 0 or (
            (('none' in dict_falls_in_question['FallacyModel/checkpoint_fallacious_phi4mini2']  or dict_falls_in_question['FallacyModel/checkpoint_fallacious_phi4mini2'] == set()) and 
            ('none' in dict_falls_in_question['FallacyModel/checkpoint_fallacious_mistral'] or dict_falls_in_question['FallacyModel/checkpoint_fallacious_mistral'] == set()))
        ):
            num_agree_phi_mistral += 1

        if len(dict_falls_in_question['FallacyModel/checkpoint_fallacious_phi4mini2'] & dict_falls_in_question['qwen_normal']) > 0 or (
            (('none' in dict_falls_in_question['FallacyModel/checkpoint_fallacious_phi4mini2'] or dict_falls_in_question['FallacyModel/checkpoint_fallacious_phi4mini2'] == set()) and 
            ('none' in dict_falls_in_question['qwen_normal'] or dict_falls_in_question['qwen_normal'] == set()))
        ):
            num_agree_phi_qwen += 1

        if len(dict_falls_in_question['FallacyModel/checkpoint_fallacious_mistral'] & dict_falls_in_question['qwen_normal']) > 0 or (
            (('none' in dict_falls_in_question['FallacyModel/checkpoint_fallacious_mistral'] or dict_falls_in_question['FallacyModel/checkpoint_fallacious_mistral'] == set()) and 
            ('none' in dict_falls_in_question['qwen_normal'] or dict_falls_in_question['qwen_normal'] == set()))
        ):
            num_agree_mistral_qwen += 1

        if len(dict_falls_in_question['FallacyFinetuning/checkpoint_fallacies'] & dict_falls_in_question['FallacyModel/checkpoint_fallacious_mistral']) > 0 or (
            (('none' in dict_falls_in_question['FallacyFinetuning/checkpoint_fallacies'] or dict_falls_in_question['FallacyFinetuning/checkpoint_fallacies'] == set()) and 
            ('none' in dict_falls_in_question['FallacyModel/checkpoint_fallacious_mistral'] or dict_falls_in_question['FallacyModel/checkpoint_fallacious_mistral'] == set()))
        ):
            num_agree_mistral24_mistral += 1

        if len(dict_falls_in_question['FallacyFinetuning/checkpoint_fallacies'] & dict_falls_in_question['FallacyModel/checkpoint_fallacious_phi4mini2']) > 0 or (
            (('none' in dict_falls_in_question['FallacyFinetuning/checkpoint_fallacies'] or dict_falls_in_question['FallacyFinetuning/checkpoint_fallacies'] == set()) and 
            ('none' in dict_falls_in_question['FallacyModel/checkpoint_fallacious_phi4mini2'] or dict_falls_in_question['FallacyModel/checkpoint_fallacious_phi4mini2'] == set()))
        ):
            num_agree_mistral24_phi += 1

        if len(dict_falls_in_question['FallacyFinetuning/checkpoint_fallacies'] & dict_falls_in_question['qwen_normal']) > 0 or (
            (('none' in dict_falls_in_question['FallacyFinetuning/checkpoint_fallacies'] or dict_falls_in_question['FallacyFinetuning/checkpoint_fallacies'] == set()) and 
            ('none' in dict_falls_in_question['qwen_normal'] or dict_falls_in_question['qwen_normal'] == set()))
        ):
            print("agreement mistral24 and qwen", question, "\nwith: ", dict_falls_in_question)
            num_agree_mistral24_qwen += 1


                
        # Compute common fallacies (fallacies that appear in all sets)
        common_fallacies = set.intersection(*dict_falls_in_question.values())
        if common_fallacies:
            num_ques_some_fallacy_common_to_all += 1
    
        # Track total number of fallacies found across all questions for each model
        num_fallacies_total_phi += len(dict_falls_in_question['FallacyModel/checkpoint_fallacious_phi4mini2'])
        num_fallacies_total_mistral += len(dict_falls_in_question['FallacyModel/checkpoint_fallacious_mistral'])
        num_fallacies_total_qwen += len(dict_falls_in_question['qwen_normal'])
        num_fallacies_total_mistral24 += len(dict_falls_in_question['FallacyFinetuning/checkpoint_fallacies'])

        # Track if a specific model found at least 1 fallacy in the current question
        if len(dict_falls_in_question['FallacyModel/checkpoint_fallacious_phi4mini2']) > 0:
            num_questions_found_fallacy_phi += 1
        if len(dict_falls_in_question['FallacyModel/checkpoint_fallacious_mistral']) > 0:
            num_questions_found_fallacy_mistral += 1
        if len(dict_falls_in_question['qwen_normal']) > 0:
            num_questions_found_fallacy_qwen += 1
        if len(dict_falls_in_question["FallacyFinetuning/checkpoint_fallacies"]) >0:
            num_questions_found_fallacy_mistral24 += 1

        shared_any = False
        for model1, model2 in [('FallacyModel/checkpoint_fallacious_phi4mini2', 'FallacyModel/checkpoint_fallacious_mistral'),
                            ('FallacyModel/checkpoint_fallacious_phi4mini2', 'qwen_normal'),
                            ('FallacyModel/checkpoint_fallacious_mistral', 'qwen_normal'),
                            ('FallacyFinetuning/checkpoint_fallacies', 'FallacyModel/checkpoint_fallacious_phi4mini2'),
                            ('FallacyFinetuning/checkpoint_fallacies', 'FallacyModel/checkpoint_fallacious_mistral'),
                            ('FallacyFinetuning/checkpoint_fallacies', 'qwen_normal')]:
            if len(dict_falls_in_question[model1] & dict_falls_in_question[model2]) > 0:
                shared_any = True
                break
        if not shared_any:
            num_ques_0_fall_common_to_any += 1

    percentage_zero_fallacies = (num_questions_with_zero_fallacies / num_questions) * 100 if num_questions > 0 else 0
    avg_fallacies_phi = num_fallacies_total_phi / num_questions if num_questions > 0 else 0
    avg_fallacies_mistral = num_fallacies_total_mistral / num_questions if num_questions > 0 else 0
    avg_fallacies_qwen = num_fallacies_total_qwen / num_questions if num_questions > 0 else 0
    avg_fallacies_mistral24 = num_fallacies_total_mistral24 / num_questions if num_questions > 0 else 0
    agreement_phi_mistral = (num_agree_phi_mistral / num_questions) * 100 if num_questions > 0 else 0
    agreement_phi_qwen = (num_agree_phi_qwen / num_questions) * 100 if num_questions > 0 else 0
    agreement_mistral_qwen = (num_agree_mistral_qwen / num_questions) * 100 if num_questions > 0 else 0
    agreement_mistral_mistral24= (num_agree_mistral24_mistral / num_questions) * 100 if num_questions > 0 else 0
    agreement_mistral24_phi = (num_agree_mistral24_phi / num_questions) * 100 if num_questions > 0 else 0
    agreement_mistral24_qwen = (num_agree_mistral24_qwen / num_questions) * 100 if num_questions > 0 else 0

    fallacy_frequency_sorted = {k: round(v, 2) for k, v in sorted(fallacy_frequency.items())}
    statistics = {
    "total_questions": round(num_questions, 2),
    "total falls": num_fallacies_total_mistral+num_fallacies_total_mistral24+num_fallacies_total_phi+num_fallacies_total_qwen,
    "questions_with_at_least_1_shared": num_ques_some_fallacy_common_to_all,
    "questions_with_no_fallacies_shared": round(num_ques_0_fall_common_to_any, 2),
    "questions_where_phi_found_fallacies": round(num_questions_found_fallacy_phi, 2),
    "questions_where_mistral_found_fallacies": round(num_questions_found_fallacy_mistral, 2),
    "questions_where_qwen_found_fallacies": round(num_questions_found_fallacy_qwen, 2),
    "questions_where_mistral24_found_fallacies": round(num_questions_found_fallacy_mistral24, 2),
    "total_fallacies_found_by_phi": round(num_fallacies_total_phi, 2),
    "total_fallacies_found_by_mistral": round(num_fallacies_total_mistral, 2),
    "total_fallacies_found_by_qwen": round(num_fallacies_total_qwen, 2),
    "total_fallacies_found_by_mistral24": round(num_fallacies_total_mistral24, 2),
    "agreement_mistral24_qwen": round(agreement_mistral24_qwen, 2),
    "agreement_mistral7_mistral24": round(agreement_mistral_mistral24, 2),
    "agreement_mistral24_phi": round(agreement_mistral24_phi, 2),
    "agreement_phi_mistral": round(agreement_phi_mistral, 2),
    "agreement_phi_qwen": round(agreement_phi_qwen, 2),
    "agreement_mistral_qwen": round(agreement_mistral_qwen, 2),
    "avg_fallacies_per_question_mistral24": round(avg_fallacies_mistral24, 2),
    "avg_fallacies_per_question_phi": round(avg_fallacies_phi, 2),
    "avg_fallacies_per_question_mistral": round(avg_fallacies_mistral, 2),
    "avg_fallacies_per_question_qwen": round(avg_fallacies_qwen, 2),
    "percentage_questions_with_zero_fallacies": round(percentage_zero_fallacies, 2),
    "fallacy_frequency_distribution": fallacy_frequency_sorted,
    }
    return statistics


def calculate_fallacy_statistics_for_all_files(combined_files, folder_mapping, detector_names, stats_output_file):
    """
    Calculate statistics for each input file and save them into a single stats file.
    
    input_files: List of input JSON files to process.
    folder_mapping: Dictionary mapping detector names to their respective folder paths.
    detector_names: List of detector names to track.
    stats_output_file: Path to save the statistics JSON file.
    """
    all_stats = {} 

    for combined_file in combined_files:
        print(f"currently computing stats for combined file {combined_file}")
        data = load_json(combined_file)

        statistics = calculate_fallacy_statistics(data, detector_names)
        all_stats[combined_file] = statistics
    
    os.makedirs(os.path.dirname(stats_output_file), exist_ok=True)
    with open(stats_output_file, "w", encoding="utf-8") as f:
        json.dump({"stats": all_stats}, f, indent=4, ensure_ascii=False)
    print(f"All statistics saved to {stats_output_file}")


def combine_fallacy_statistics(json_objects, outfile):
    # Initialize accumulators
    num_questions = 0
    num_fallacies_total_phi = 0
    num_fallacies_total_mistral = 0
    num_fallacies_total_qwen = 0
    num_fallacies_total_mistral24 = 0
    num_ques_some_fallacy_common_to_all = 0
    num_ques_0_fall_common_to_any = 0
    num_questions_found_fallacy_phi = 0
    num_questions_found_fallacy_mistral = 0
    num_questions_found_fallacy_qwen = 0
    num_questions_found_fallacy_mistral24 = 0
    agreement_mistral24_qwen = 0
    agreement_mistral_mistral24 = 0
    agreement_mistral24_phi = 0
    agreement_phi_mistral = 0
    agreement_phi_qwen = 0
    agreement_mistral_qwen = 0
    avg_fallacies_mistral24 = 0
    avg_fallacies_phi = 0
    avg_fallacies_mistral = 0
    avg_fallacies_qwen = 0
    percentage_zero_fallacies = 0
    fallacy_frequency = defaultdict(int)

    num_files = len(json_objects)

    # Process each JSON object
    for data in json_objects:
        stats = data["stats"]["results/combined_files/combined_phi4-mini_4bit_all_short.json"]

        num_questions += stats["total_questions"]
        num_fallacies_total_phi += stats["total_fallacies_found_by_phi"]
        num_fallacies_total_mistral += stats["total_fallacies_found_by_mistral"]
        num_fallacies_total_qwen += stats["total_fallacies_found_by_qwen"]
        num_fallacies_total_mistral24 += stats["total_fallacies_found_by_mistral24"]
        num_ques_some_fallacy_common_to_all += stats["questions_with_at_least_1_shared"]
        num_ques_0_fall_common_to_any += stats["questions_with_no_fallacies_shared"]
        num_questions_found_fallacy_phi += stats["questions_where_phi_found_fallacies"]
        num_questions_found_fallacy_mistral += stats["questions_where_mistral_found_fallacies"]
        num_questions_found_fallacy_qwen += stats["questions_where_qwen_found_fallacies"]
        num_questions_found_fallacy_mistral24 += stats["questions_where_mistral24_found_fallacies"]

        agreement_mistral24_qwen += stats["agreement_mistral24_qwen"]
        agreement_mistral_mistral24 += stats["agreement_mistral7_mistral24"]
        agreement_mistral24_phi += stats["agreement_mistral24_phi"]
        agreement_phi_mistral += stats["agreement_phi_mistral"]
        agreement_phi_qwen += stats["agreement_phi_qwen"]
        agreement_mistral_qwen += stats["agreement_mistral_qwen"]

        avg_fallacies_mistral24 += stats["avg_fallacies_per_question_mistral24"]
        avg_fallacies_phi += stats["avg_fallacies_per_question_phi"]
        avg_fallacies_mistral += stats["avg_fallacies_per_question_mistral"]
        avg_fallacies_qwen += stats["avg_fallacies_per_question_qwen"]

        percentage_zero_fallacies += stats["percentage_questions_with_zero_fallacies"]

        for fallacy, count in stats["fallacy_frequency_distribution"].items():
            fallacy_frequency[fallacy] += count

    # Compute averages
    avg_fallacies_mistral24 /= num_files
    avg_fallacies_phi /= num_files
    avg_fallacies_mistral /= num_files
    avg_fallacies_qwen /= num_files
    percentage_zero_fallacies /= num_files
    agreement_mistral24_qwen /= num_files
    agreement_mistral_mistral24 /= num_files
    agreement_mistral24_phi /= num_files
    agreement_phi_mistral /= num_files
    agreement_phi_qwen /= num_files
    agreement_mistral_qwen /= num_files

    # Sort fallacy frequency
    fallacy_frequency_sorted = dict(sorted(fallacy_frequency.items(), key=lambda item: item[1], reverse=True))

    # Build final statistics dictionary
    statistics = {
        "total_questions": round(num_questions, 2),
        "total falls": num_fallacies_total_mistral + num_fallacies_total_mistral24 + num_fallacies_total_phi + num_fallacies_total_qwen,
        "questions_with_at_least_1_shared": num_ques_some_fallacy_common_to_all,
        "questions_with_no_fallacies_shared": round(num_ques_0_fall_common_to_any, 2),
        "questions_where_phi_found_fallacies": round(num_questions_found_fallacy_phi, 2),
        "questions_where_mistral_found_fallacies": round(num_questions_found_fallacy_mistral, 2),
        "questions_where_qwen_found_fallacies": round(num_questions_found_fallacy_qwen, 2),
        "questions_where_mistral24_found_fallacies": round(num_questions_found_fallacy_mistral24, 2),
        "total_fallacies_found_by_phi": round(num_fallacies_total_phi, 2),
        "total_fallacies_found_by_mistral": round(num_fallacies_total_mistral, 2),
        "total_fallacies_found_by_qwen": round(num_fallacies_total_qwen, 2),
        "total_fallacies_found_by_mistral24": round(num_fallacies_total_mistral24, 2),
        "agreement_mistral24_qwen": round(agreement_mistral24_qwen, 2),
        "agreement_mistral7_mistral24": round(agreement_mistral_mistral24, 2),
        "agreement_mistral24_phi": round(agreement_mistral24_phi, 2),
        "agreement_phi_mistral": round(agreement_phi_mistral, 2),
        "agreement_phi_qwen": round(agreement_phi_qwen, 2),
        "agreement_mistral_qwen": round(agreement_mistral_qwen, 2),
        "avg_fallacies_per_question_mistral24": round(avg_fallacies_mistral24, 2),
        "avg_fallacies_per_question_phi": round(avg_fallacies_phi, 2),
        "avg_fallacies_per_question_mistral": round(avg_fallacies_mistral, 2),
        "avg_fallacies_per_question_qwen": round(avg_fallacies_qwen, 2),
        "percentage_questions_with_zero_fallacies": round(percentage_zero_fallacies, 2),
        "fallacy_frequency_distribution": fallacy_frequency_sorted,
    }


    # Example usage
    json_files = [json_data1, json_data2]  # List of JSON objects
    combined_statistics = statistics
    print(json.dumps(combined_statistics, indent=4))

######################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine fallacy detections from multiple models into one JSON file.")
    parser.add_argument("--input_files", type=str, nargs='+', required=True, help="List of input JSON files to process.")
    parser.add_argument("--stats_name", type=str, required=True, help="Output statistics file name.")
    args = parser.parse_args()

    # Mapping of fallacy detector model folders
    folder_mapping = {
        "phi4-mini_fallacious": "results/tests_on_all_fallacious_phi4mini2",
        "mistral_fallacious": "results/tests_on_all_fallacious_mistral",
        "mistral24b_fallacious": "results/tests_on_all_fallacious_mistral24B",
        "qwen_fallacious": "results/tests_on_all_fallacious_qwen"
    }

    # Detector names to track in statistics
    detector_names = ["FallacyModel/checkpoint_fallacious_phi4mini2", "FallacyModel/checkpoint_fallacious_mistral", "qwen_normal", "FallacyFinetuning/checkpoint_fallacies"]
    files_outputted = []
    # Process each input file to combine fallacy detections
    for input_file in args.input_files:
        output_file = f"results/combined_files/combined_{input_file}"
        files_outputted.append(output_file)
        print(f"Processing file: {input_file}")
        combine_fallacy_detections(input_file, output_file, folder_mapping, detector_names)

    # Calculate and save statistics after all files have been processed
    stats_output_file = f"results/combined_files/stats_files/{args.stats_name}"
    calculate_fallacy_statistics_for_all_files(files_outputted, folder_mapping, detector_names, stats_output_file)
