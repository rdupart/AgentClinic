# -*- coding: utf-8 -*-

import openai
import json

with open("AgentClinic-main/agentclinic_medqa.jsonl", "r") as f:
    # Load the first example only
    first_line = f.readline()
    case_data = json.loads(first_line)
    #test_data = case_data['Correct_Diagnosis']  #[json.loads(line) for line in f]


class PatientAgent:
    def __init__(self, case_data):
        self.case = case_data

    def present_case(self):
        osce = self.case["OSCE_Examination"]
        patient = osce["Patient_Actor"]
        exam = osce["Physical_Examination_Findings"]
        tests = osce["Test_Results"]

        return f"""Patient Presentation:
Demographics: {patient['Demographics']}
History: {patient['History']}
Primary Symptom: {patient['Symptoms']['Primary_Symptom']}
Secondary Symptoms: {', '.join(patient['Symptoms']['Secondary_Symptoms'])}
Past Medical History: {patient['Past_Medical_History']}
Social History: {patient['Social_History']}
Review of Systems: {patient['Review_of_Systems']}

Physical Exam:
Vitals: {exam['Vital_Signs']}
Neurologic: {exam['Neurological_Examination']}

Test Results:
Blood Tests: {tests['Blood_Tests']}
Electromyography: {tests['Electromyography']}
Imaging: {tests['Imaging']}
"""

patient = PatientAgent(case_data)
initial_prompt = patient.present_case()
print(initial_prompt)

#patient = PatientAgent(case_data)
#initial_prompt = patient.present_case()
#print(initial_prompt)

class DoctorAgent:
    def __init__(self, role="General Physician"):
        self.role = role
        self.system_prompt = f"You are a {self.role} tasked with diagnosing complex medical cases. Focus only on your specialty. Ask for lab/image interpretation if needed."

    def query(self, patient_prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": patient_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"[DoctorAgent Error]: {e}"

from PIL import Image
import os

class MeasurementAgent:
    def __init__(self):
        self.system_prompt = (
            "You are a medical diagnostics specialist. Analyze and interpret lab tests, EMG, or medical images "
            "to support a physician's diagnosis. Respond clearly with structured findings."
        )

    def query(self, test_type, test_input):
        try:
            # If image path exists, handle as VLM input
            if test_type == "image" and os.path.exists(test_input):
                with open(test_input, "rb") as img_file:
                    image_bytes = img_file.read()
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": "Please interpret this medical image."}
                    ],
                    temperature=0.2,
                    max_tokens=500,
                    images=[{"image": image_bytes}]
                )
            else:
                # Default to LLM for lab/EMG interpretation
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Interpret the following results:\n{test_input}"}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"[MeasurementAgent Error]: {e}"

class SpecialistAgent:
    def __init__(self, role, knowledge_base=None, instructions=None):
        self.role = role
        self.knowledge_base = knowledge_base if knowledge_base else {}
        self.instructions = instructions if instructions else {}
        self.final_decision = False
        self.system_prompt = (
            f"You are a {self.role} providing a second opinion on a diagnostic case. "
            f"Comment strictly within your specialty and support or refute the proposed diagnosis.\n"
            f"Specialty knowledge: {self.knowledge_base}\n"
            f"Instructions: {self.instructions}"
        )

    def query(self, patient_prompt, doctor_analysis, round_type="initial"):
        """
        round_type: 'initial' for the first input, 'final' for final decision round
        """
        try:
            if round_type == "final":
                self.final_decision = True

            full_prompt = f"""
The following is a medical case:

{patient_prompt}

The primary doctor (specialty: main doctor) has proposed the following analysis:

{doctor_analysis}

As a {self.role}, provide your feedback on the case. Focus on whether you agree, and if not, suggest alternatives.
"""
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3,
                max_tokens=700
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"[{self.role} Error]: {e}"

def feedback_loop(specialists, patient_prompt, doctor_opinion):
    # Step 1: Get initial opinions from specialists
    initial_opinions = {}
    for specialist in specialists:
        initial_opinions[specialist.role] = specialist.query(patient_prompt, doctor_opinion, round_type="initial")
    
    # Step 2: Simulate the feedback phase where specialists hear each other's input
    feedback = ""
    for role, opinion in initial_opinions.items():
        feedback += f"{role} opinion:\n{opinion}\n\n"
    
    # Step 3: Specialists hear each other's feedback and adjust their reasoning
    updated_opinions = {}
    for specialist in specialists:
        updated_opinions[specialist.role] = specialist.query(patient_prompt, feedback, round_type="final")
    
    return updated_opinions

def final_decision(specialists, patient_prompt, doctor_opinion, final_feedback):
    # Step 4: Get final decisions based on feedback and reasoning
    final_decisions = {}
    for specialist in specialists:
        final_decisions[specialist.role] = specialist.query(patient_prompt, final_feedback, round_type="final")
    
    return final_decisions

class ModeratorAgent:
    def __init__(self):
        self.system_prompt = (
            "You are a senior medical moderator. Given the dialogue and final diagnosis from two teams of doctors, "
            "determine if the diagnosis is valid. If so, summarize. If not, explain what’s missing and recommend a restart."
        )

    def evaluate(self, case_summary):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": case_summary}
                ],
                temperature=0.2,
                max_tokens=700
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"[ModeratorAgent Error]: {e}"

class AgentnosisCore:
    def __init__(self, case_data):
        self.case_data = case_data
        self.patient_agent = PatientAgent(case_data)
        self.doctor_agent = DoctorAgent(role="Neurologist")  # example specialty
        self.measurement_agent = MeasurementAgent()

        # Adding Cardiologist Specialist with domain knowledge
        cardiologist_knowledge = {
            "heart_conditions": ["Coronary Artery Disease", "Heart Failure", "Arrhythmias"],
            "diagnostic_tests": ["ECG", "Stress test", "Angiogram"],
            "treatments": ["Beta-blockers", "ACE inhibitors", "Statins", "Angioplasty"]
        }

        cardiologist_instructions = {
            "specialization": "Diagnose and treat cardiovascular issues like coronary artery disease, arrhythmias, etc.",
            "patient_interaction": "Ask questions about chest pain, shortness of breath, dizziness, and prior medical history related to heart conditions.",
            "diagnostic_process": "Based on symptoms and diagnostic tests, provide a treatment plan. Consult other specialists as necessary."
        }

        self.specialist1 = SpecialistAgent(role="Cardiologist", knowledge_base=cardiologist_knowledge, instructions=cardiologist_instructions)
        self.specialist2 = SpecialistAgent(role="Rheumatologist")    # Secondary
        self.specialist3 = SpecialistAgent(role="Pharmacologist")    # Secondary
        self.moderator_agent = ModeratorAgent()

    def run_case(self):
        # STEP 1: Present case
        patient_prompt = self.patient_agent.present_case()

        # STEP 2: Doctor's Initial Analysis
        doctor_opinion = self.doctor_agent.query(patient_prompt)

        # STEP 3: Measurement Agent (lab + emg only for now)
        labs = self.case_data["OSCE_Examination"]["Test_Results"]["Blood_Tests"]
        emg = self.case_data["OSCE_Examination"]["Test_Results"]["Electromyography"]

        lab_analysis = self.measurement_agent.query("lab", str(labs))
        emg_analysis = self.measurement_agent.query("lab", str(emg))

        full_case_prompt = f"{patient_prompt}\n\nLab Analysis:\n{lab_analysis}\n\nEMG Analysis:\n{emg_analysis}"

        # STEP 4: Collect Specialists' Initial Feedback
        specialists = [self.specialist1, self.specialist2, self.specialist3]
        updated_opinions = feedback_loop(specialists, patient_prompt, doctor_opinion)

        # STEP 5: Final Decision Round (Private Decisions)
        final_feedback = "\n".join([f"{role}: {opinion}" for role, opinion in updated_opinions.items()])
        final_decisions = final_decision(specialists, patient_prompt, doctor_opinion, final_feedback)

        # STEP 6: Moderator Evaluation
        moderator_feedback = self.moderator_agent.evaluate(final_feedback)

        # Final Output
        return {
            "Patient Prompt": patient_prompt,
            "Lab + EMG Interpretation": f"{lab_analysis}\n\n{emg_analysis}",
            "Doctor Opinion": doctor_opinion,
            "Specialist1 Opinion (Cardiologist)": final_decisions["Cardiologist"],
            "Specialist2 Opinion (Rheumatologist)": final_decisions["Rheumatologist"],
            "Specialist3 Opinion (Pharmacologist)": final_decisions["Pharmacologist"],
            "Moderator Feedback": moderator_feedback
        }

agentnosis = AgentnosisCore(case_data)
results = agentnosis.run_case()

def compare_results(diagnosis, correct_diagnosis):
    """
    Compare the diagnosis given by the agent with the correct diagnosis from the scenario.
    """
    return diagnosis.lower() == correct_diagnosis.lower()

# Get the correct diagnosis for the first case
correct_diagnosis = case_data["OSCE_Examination"]["Correct_Diagnosis"]

# Run the accuracy test using the first case
final_diagnosis = results["Doctor Opinion"]  # For simplicity, we use the doctor's opinion here

# Calculate accuracy
is_correct = compare_results(final_diagnosis, correct_diagnosis)

# Print the result (accuracy)
if is_correct:
    accuracy = 100  # If correct diagnosis
    print(f"Accuracy: {accuracy}% - Diagnosis is correct.")
else:
    accuracy = 0  # If diagnosis is incorrect
    print(f"Accuracy: {accuracy}% - Diagnosis is incorrect. Correct diagnosis: {correct_diagnosis}, Agent's diagnosis: {final_diagnosis}")

# Display all results
for section, content in results.items():
    print(f"\n\n===== {section} =====\n{content}\n")