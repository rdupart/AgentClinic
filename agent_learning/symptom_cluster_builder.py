import json
import os
import uuid  # For generating unique IDs
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN  # Changed to DBSCAN for dynamic clustering
import matplotlib.pyplot as plt
import openai


def load_symptoms_from_multiple_files(file_paths):
    symptoms_list = []
    case_ids = []

    # Check if all file paths are valid
    for file_path in file_paths:
        # Ensure the file path is correct and exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist!")
            continue  # Skip to next file

        print(f"Loading data from {file_path}")

        with open(file_path, "r") as f:
            scenario_strs = [json.loads(line) for line in f]

            # Process each scenario in the file
            for scenario in scenario_strs:
                case_id = str(uuid.uuid4())  # Generate a unique Case_ID using uuid
                symptoms = None

                # Handle the case where the 'OSCE_Examination' key might not exist
                if 'OSCE_Examination' in scenario:
                    # Standard structure for files with 'OSCE_Examination'
                    patient_actor = scenario["OSCE_Examination"].get("Patient_Actor", {})
                    primary_symptom = patient_actor.get("Symptoms", {}).get("Primary_Symptom", "No primary symptom")
                    secondary_symptoms = patient_actor.get("Symptoms", {}).get("Secondary_Symptoms", [])
                    symptoms = primary_symptom + " " + " ".join(secondary_symptoms)
                elif 'patient_info' in scenario:
                    # Custom structure for the NEJM file (or any other with a similar format)
                    symptoms = scenario.get("patient_info", "No symptoms found")

                # Add the newly generated Case_ID
                case_ids.append(case_id)
                symptoms_list.append(symptoms)

                # Optionally, you can update the scenario to include this new Case_ID
                if 'OSCE_Examination' in scenario:
                    scenario["OSCE_Examination"]["Case_ID"] = case_id
                elif 'question' in scenario:  # for NEJM style, add the Case_ID here
                    scenario["Case_ID"] = case_id

    print(f"Loaded {len(symptoms_list)} symptoms from {len(file_paths)} files.")
    return symptoms_list, case_ids

def embed_symptoms(symptoms_list):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(symptoms_list)  # Now symptoms_list is a list of strings
    return embeddings

# Use DBSCAN instead of KMeans
def cluster_embeddings_with_dbscan(embeddings, eps=0.5, min_samples=5):
    """
    Perform clustering with DBSCAN without needing to specify the number of clusters.
    - `eps` is the maximum distance between two samples to be considered as in the same neighborhood.
    - `min_samples` is the number of samples in a neighborhood for a point to be considered as a core point.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = db.fit_predict(embeddings)
    return cluster_labels

def save_cluster_mapping(cluster_to_specialists, save_path="symptom_cluster_mapping.json"):
    # Convert numpy.int64 keys to int
    cluster_to_specialists_fixed = {int(k): v for k, v in cluster_to_specialists.items()}
    
    with open(save_path, "w") as f:
        json.dump(cluster_to_specialists_fixed, f, indent=2)

def auto_assign_specialists_gpt(symptoms_list):
    """
    Given a list of symptom strings for a cluster,
    query GPT-4 to suggest relevant medical specialists.
    """
    limited_symptoms = symptoms_list[:15]
    combined_symptoms = "\n".join(limited_symptoms)

    prompt = (
        "You are a medical expert. Given the following patient symptoms, "
        "please list the most relevant medical specialists who should handle these cases.\n\n"
        f"Symptoms:\n{combined_symptoms}\n\n"
        "Respond with a comma-separated list of specialists only."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    specialists_text = response['choices'][0]['message']['content']
    specialists = [s.strip() for s in specialists_text.split(",")]
    return specialists

def main():
    cwd = os.getcwd()
    print(f"Current Working Directory: {cwd}")  # Debugging current working directory

    # Provide the correct paths to the files based on the current directory
    file_paths = [
        os.path.join(cwd, "agentclinic_medqa.jsonl"),
        os.path.join(cwd, "agentclinic_medqa_extended.jsonl"),
        os.path.join(cwd, "agentclinic_mimiciv.jsonl"),
        os.path.join(cwd, "agentclinic_nejm.jsonl"),
        os.path.join(cwd, "agentclinic_nejm_extended.jsonl")
    ]

    # 1. Load symptoms from all files
    symptoms_list, case_ids = load_symptoms_from_multiple_files(file_paths)
    print(f"Loaded {len(symptoms_list)} cases from {len(file_paths)} files.")
    
    if len(symptoms_list) == 0:
        print("No symptoms data loaded. Please check your file paths.")
        return  # Exit if no data is loaded

    # 2. Embed symptoms using SentenceTransformers
    embeddings = embed_symptoms(symptoms_list)
    print(f"Generated {len(embeddings)} embeddings.")

    # 3. Cluster symptoms into groups using DBSCAN (dynamic clustering)
    cluster_labels = cluster_embeddings_with_dbscan(embeddings)
    print(f"Clustered into {len(set(cluster_labels))} groups (including noise).")

    # 4. Show example cases per cluster
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((case_ids[idx], symptoms_list[idx]))  # Using idx here instead of case_id

    # Display all cases per cluster
    for cluster_id, cases in clusters.items():
        if cluster_id == -1:
            print(f"\n=== Noise (Cluster {cluster_id}) ===")
        else:
            print(f"\n=== Cluster {cluster_id} ===")
        for case_id, symptoms in cases:  # Show all cases in the cluster
            print(f"[Case {case_id}] {symptoms}")
        
    # 5. Manually input specialist assignments
    print("\nAutomatically assigning specialists to each cluster using GPT...\n")
    cluster_to_specialists = {}
    for cluster_id, cases in clusters.items():
        if cluster_id == -1:
            continue  # Skip noise cluster (-1)
        symptoms_for_cluster = [symptoms for _, symptoms in cases]
        specialists = auto_assign_specialists_gpt(symptoms_for_cluster)
        print(f"Cluster {cluster_id}: Suggested specialists: {specialists}")
        cluster_to_specialists[cluster_id] = specialists
    # 6. Save the mapping to a file
    save_cluster_mapping(cluster_to_specialists)
    print("\n✅ Saved mapping to 'symptom_cluster_mapping.json'.")

if __name__ == "__main__":
    main()
