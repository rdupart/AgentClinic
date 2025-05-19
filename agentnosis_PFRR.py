import json
import argparse
import anthropic
#from transformers import pipeline
import openai, re, random, time, json, replicate, os

total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

def call_openai_chat(messages, **kwargs):
    global total_prompt_tokens, total_completion_tokens, total_tokens
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        **kwargs
    )
    usage = response.get('usage', {})
    total_prompt_tokens += usage.get('prompt_tokens', 0)
    total_completion_tokens += usage.get('completion_tokens', 0)
    total_tokens += usage.get('total_tokens', 0)

    return response['choices'][0]['message']['content']

def query_model(backend, prompt, system_prompt="", image_requested=False, scene=None):
    global total_prompt_tokens, total_completion_tokens, total_tokens
    if backend.startswith("gpt"):
        response = openai.ChatCompletion.create(
            model="gpt-4o" if backend == "gpt4" else backend,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        usage = response.get('usage', {})
        total_prompt_tokens += usage.get('prompt_tokens', 0)
        total_completion_tokens += usage.get('completion_tokens', 0)
        total_tokens += usage.get('total_tokens', 0)
        return response['choices'][0]['message']['content']
    else:
        raise NotImplementedError(f"Model backend {backend} not supported yet.")


class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
   
    def patient_information(self) -> dict:
        return self.patient_info




    def examiner_information(self) -> dict:
        return self.examiner_info
   
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
   
    def diagnosis_information(self) -> dict:
        return self.diagnosis



class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("agentclinic_medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
   
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
   
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
       


class ScenarioMedQAExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
   
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
   
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
   
    def diagnosis_information(self) -> dict:
        return self.diagnosis








class ScenarioLoaderMedQAExtended:
    def __init__(self) -> None:
        with open("agentclinic_medqa_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQAExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
   
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
   
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
       








class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
   
    def patient_information(self) -> dict:
        return self.patient_info




    def examiner_information(self) -> dict:
        return self.examiner_info
   
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
   
    def diagnosis_information(self) -> dict:
        return self.diagnosis








class ScenarioLoaderMIMICIV:
    def __init__(self) -> None:
        with open("agentclinic_mimiciv.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
   
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
   
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]








class ScenarioNEJMExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.question = scenario_dict["question"]
        self.image_url = scenario_dict["image_url"]
        self.diagnosis = [_sd["text"]
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]




    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info




    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
   
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
   
    def diagnosis_information(self) -> str:
        return self.diagnosis




class ScenarioLoaderNEJMExtended:
    def __init__(self) -> None:
        with open("agentclinic_nejm_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJMExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
   
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
   
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]



class ScenarioNEJM:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.question = scenario_dict["question"]
        self.image_url = scenario_dict["image_url"]
        self.diagnosis = [_sd["text"]
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]




    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info




    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
   
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
   
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJM:
    def __init__(self) -> None:
        with open("agentclinic_nejm.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJM(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
   
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
   
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()
        self.pipe = None




        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]




    def generate_bias(self) -> str:
        """
        ================
        Cognitive biases
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""




    def inference_patient(self, question) -> str:
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: ", self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer




    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + bias_prompt + symptoms
   
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()




    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=20, bias_present=None, img_request=False) -> None:
        # number of inference calls to the doctor
        self.infs = 0
        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for doctor
        self.presentation = ""
        # language model backend for doctor agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.vital_signs = self.scenario.physical_exams.get("Vital_Signs", {})  
        self.reset()
        self.pipe = None
        self.img_request = img_request
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]
       




    def generate_bias(self) -> str:
        """
        ================
        Cognitive biases
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""




    def inference_doctor(self, question, image_requested=False) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS:
            # If the maximum number of inferences is reached, generate the diagnosis
            return self.generate_diagnosis()
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: ", self.system_prompt(), image_requested=image_requested, scene=self.scenario)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer




    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = (
            "You are a doctor named Dr. Agent who only responds in the form of dialogue. "
            "You are inspecting a patient and will ask questions to understand their disease. "
            "You are only allowed to ask {} questions total before you must make a decision. "
            "You have asked {} questions so far. You can only request *vital signs* that are available in the scenario using the format \"REQUEST TEST: [test]\". "
            "For example, \"REQUEST TEST: Heart Rate\". "
            "If a test is not available or is rejected, continue reasoning with the information you already have. "
            "Do not stop or wait for a test — aim to reach a diagnosis using all clues provided."
            "Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis, please type \"DIAGNOSIS READY: [diagnosis here]\".".format(self.MAX_INFS, self.infs)
        )
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
        return base + bias_prompt + presentation




    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()




    def generate_diagnosis(self) -> str:
            """
            This is the core function where GPT-4 generates a diagnosis based on the full dialogue history
            """
            prompt = f"Based on the following patient history, please provide the most likely diagnosis:\n\n{self.agent_hist}\n\nDiagnosis: "
            diagnosis = query_model(self.backend, prompt, self.system_prompt())
            return diagnosis.strip()  # Ensure to clean up the result

class SpecialistAgent_AsDoctor:
    def __init__(self, specialty, allowed_images, allowed_tests, scenario, backend_str="gpt4", max_infs=20, bias_present=None, img_request=False) -> None:
        self.specialty = specialty  # e.g. "Neurologist"
        # number of inference calls to the doctor
        self.infs = 0
        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for doctor
        self.presentation = ""
        self.allowed_tests = allowed_tests
        # language model backend for doctor agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.allowed_images = allowed_images  # e.g. ["MRI Brain"]
        self.images_requested = 0
        self.max_images_allowed = 2

        self.tests_requested = 0  # Track number of tests requested
        #self.allowed_tests = self.get_relevant_tests(specialty)
        self.reset()
        self.pipe = None
        self.img_request = img_request       



    def inference_specialist(self, question, image_requested=False) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS:
            # If the maximum number of inferences is reached, generate the diagnosis
            return self.generate_diagnosis()
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: ", self.system_prompt(), image_requested=image_requested, scene=self.scenario)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()

        # Ask model to focus on the specialty dynamically, without hard-coding each one
        dynamic_focus = (
            f"You are a {self.specialty} doctor named Dr. Agent. "
            "You may only request tests that are directly relevant to your specialty, to diagnose the patient's condition. Do not request general, unrelated, or duplicate tests."
            "Tailor your questions to focus on symptoms, signs, and tests relevant to your specialty. "
            "Avoid asking generic or irrelevant questions outside your specialty. "
        )

        base = (
            dynamic_focus +
            "You are allowed to ask one question at a time. Do not list multiple questions in one turn."
            "Ask concise questions (1-3 sentences). "
            "You can request tests with 'REQUEST TEST: [test]'. "
            "When ready to diagnose, type 'DIAGNOSIS READY: [diagnosis]'."
        )
        allowed_tests_str = ", ".join(self.allowed_tests)

        test_restriction = (
            f"\n\nYou are only allowed to request the following tests:\n"
            f"{allowed_tests_str}\n"
            "If a test is not on this list, do not request it."
            " If a test or image request is rejected, continue reasoning based on the information you already have. "
            " Do not repeat the same request or stop asking questions. Work toward a diagnosis with available data."

        )

        if self.allowed_images:
            allowed_img_str = ", ".join(self.allowed_images)
            base += f" You may request the following medical images using 'REQUEST IMAGE: [image]': {allowed_img_str}. You are limited to {self.max_images_allowed} image(s)."

        #if self.img_request:
            #base += " You may also request medical images related to your specialty."

        presentation = f"\n\nPatient information and history:\n{self.presentation}\n"

        return base + test_restriction + bias_prompt + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()

    def generate_diagnosis(self) -> str:
            """
            This is the core function where GPT-4 generates a diagnosis based on the full dialogue history
            """
            prompt = f"Based on the following patient history, please provide the most likely diagnosis:\n\n{self.agent_hist}\n\nDiagnosis: "
            diagnosis = query_model(self.backend, prompt, self.system_prompt())
            return diagnosis.strip()  


def generate_doctor_report(doctor_agent, patient_agent, mode="medical_report"):
    """
    Generate a doctor's report based on dialogue history between doctor and patient.

    Args:
        doctor_agent: the DoctorAgent instance
        patient_agent: the PatientAgent instance
        mode (str): 'medical_report' for full report, 'doctor_report' for limited/subjective report

    Returns:
        str: formatted report string based on dialogue history
    """

    if mode == "medical_report":
        report = (
            "Report 1 - Summary from Doctor based on dialogue history (medical_report):\n\n"
            "Doctor's dialogue history:\n"
            f"{doctor_agent.agent_hist}\n\n"
            "Patient's dialogue history:\n"
            f"{patient_agent.agent_hist}\n"
        )
    elif mode == "doctor_report":
        # Possibly a shorter or subjective report: here just a placeholder
        report = (
            "Report 1 - Summary from Doctor (doctor_report):\n"
            f"Dialogue history excerpt:\n{doctor_agent.agent_hist[-1000:]}"  # last 1000 chars as snippet
        )
    else:
        raise ValueError("Invalid mode. Use 'medical_report' or 'doctor_report'.")

    return report



class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt4") -> None:
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for measurement
        self.presentation = ""
        # language model backend for measurement agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.pipe = None
        self.reset()




    def inference_measurement(self, question) -> str:
        answer = str()
        answer = query_model(self.backend, "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer




    def system_prompt(self) -> str:
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + presentation
   
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"




    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()


class SpecialistAgent:
    def __init__(self, specialty, scenario, medical_report=None, backend="gpt4"):
        self.specialty = specialty
        self.scenario = scenario
        self.backend = backend
        self.agent_hist = ""  
        self.medical_report = medical_report

    def add_to_history(self, message: str):
        self.agent_hist += f"{message}\n"


    CRITERIA_TEXT = """
    Criteria for Specialist Discussion:


    Individual Reasoning Evaluation:
    - Did you consider different possibilities?
    - Did you prioritize your logic by considering the full clinical picture (not just common patterns)?
    - How did you rule out other possibilities? How are you ruling in possibilities? (e.g., based on findings)
    - Did you consider that the patient’s symptoms might not fit the common or classic pattern of a disease?
    - Does the diagnosis make sense given the disease process?


    Team Diagnostic Process:
    - Did you incorporate input from at least two additional specialists?
    - Were conflicting findings resolved, rather than ignored?
    - Was input from all members equally valued?
    - Was consensus reached? If not, resolve disagreements in the disagreement rounds.


    Disagreement Round:
    - Is dissenting diagnosis reasoning clearly articulated?
    - Do you agree on some points? If not, why?
    - Can both viewpoints be true?
    - Does one view prove a better fit?
    - Which diagnosis is more likely?
    """


    def inference_specialist(self, dialogue_so_far, current_speaker, other_speaker):
        context_to_use = self.medical_report if self.medical_report else dialogue_so_far
        prompt = (
            f"You are the {current_speaker}, engaged in a detailed, reflective discussion with {other_speaker} "
            "about a complex medical case. The goal is to reason through possible diagnoses thoughtfully.\n\n"
            "Here is the case information from a doctor-patient interaction:\n"
            f"{context_to_use}\n\n"
            f"You can refer to the following criteria to guide your reasoning if needed:\n"
            f"{self.CRITERIA_TEXT}\n\n"
            f"Your task:\n"
            "- Respond to {other_speaker}'s last point by agreeing or disagreeing.\n"
            "- If you disagree or have alternatives, clearly state them with reasons.\n"
            "- Include a reflective question if relevant, but do not answer your own question.\n"
            "- Your response should end with a conclusion, suggestion, or next step — not just a question.\n"
            "- Your response should be concise (1-3 sentences), thoughtful, and focused on reasoning.\n\n"
            "- If you feel the discussion is complete, you may conclude with a final statement."
            f"{current_speaker}:"
        )
       
        response = query_model(self.backend, prompt)
        self.add_to_history(f"{current_speaker}: {response}")
        return response




def get_specialist_names_by_diagnosis(correct_diagnosis):
    prompt = f"""
    You are a medical expert. Based on the diagnosis below, provide a simple, comma-separated list of the MOST RELEVANT medical specialists typically involved in diagnosing, treating, or managing this condition.
    Only list the specialist names, no explanations or extra text. At least three specialists must be assigned.

    Diagnosis:
    {correct_diagnosis}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.3,
    )

    specialists_text = response['choices'][0]['message']['content'].strip()
    specialists_list = [s.strip() for s in specialists_text.split(",") if s.strip()]
    return specialists_list

def assign_specialists_and_explain_by_diagnosis(scenario):
    correct_diagnosis = scenario.diagnosis_information()  # Get correct diagnosis from scenario
    names = get_specialist_names_by_diagnosis(correct_diagnosis)

    print("Specialist names based on diagnosis:", names)
    return names


#def assign_group_specialists_by_diagnosis(scenario):
    correct_diagnosis = scenario.diagnosis_information()
    prompt = f"""
    You are a medical expert. Based on the diagnosis below, return a comma-separated list of at least FIVE relevant specialists who should be involved.
    Do NOT add explanations. Just return the specialist names.

    Diagnosis:
    {correct_diagnosis}
    """

    response = call_openai_chat(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.3,
    )

    specialists_text = response['choices'][0]['message']['content'].strip()
    specialists_list = [s.strip() for s in specialists_text.split(",") if s.strip()]
    print("Group mode specialist list:", specialists_list)
    return specialists_list

def assign_tests_to_specialists(specialists, available_tests, diagnosis):
    prompt = f"""
You are a medical expert.

Given the diagnosis: **{diagnosis}**

Here is a list of specialists involved:
{', '.join(specialists)}

And here is a list of available tests for the case:
{', '.join(available_tests)}

Assign each test to the most appropriate specialist based on medical relevance.
Return a mapping in JSON format like:
{{
    "Neurologist": ["Electromyography"],
    "Immunologist": ["Acetylcholine_Receptor_Antibodies"],
    ...
}}

Only assign each test to one specialist.
Only use the specialists provided.
Do not include explanations.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.2,
    )
    
    content = response['choices'][0]['message']['content']
    try:
        test_map = json.loads(content)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from GPT-4:", content)
        raise e

    return test_map

def extract_test_names(test_dict):
    test_names = []

    def recursive_extract(d):
        if isinstance(d, dict):
            for key, value in d.items():
                test_names.append(key.strip())
                if isinstance(value, dict):
                    recursive_extract(value)
                elif isinstance(value, str):
                    test_names.append(value.strip())
        elif isinstance(d, list):
            for item in d:
                recursive_extract(item)

    recursive_extract(test_dict)
    return list(set(test_names))

def process_scenario_assign_tests(scenario):
    diagnosis = scenario.diagnosis_information()
    test_dict = scenario.exam_information()["tests"]
    
    available_tests = extract_test_names(test_dict)
    
    specialists = assign_specialists_and_explain_by_diagnosis(scenario)
    
    assigned_map = assign_tests_to_specialists(specialists, available_tests, diagnosis)

    return assigned_map

def extract_image_names(scenario):
    """
    Extract image names from scenario.tests["Imaging"]
    Example output: ["Chest_CT", "MRI_Brain"]
    """
    imaging_dict = scenario.tests.get("Imaging", {})
    return [key.replace("_", " ") for key in imaging_dict.keys()]

def assign_images_to_specialists(specialists, available_images, diagnosis):
    prompt = f"""
You are a medical expert.

Diagnosis: {diagnosis}

Here is a list of available medical images:
{', '.join(available_images)}

And here are the specialists:
{', '.join(specialists)}

Assign each image to ONE specialist who is most relevant for interpreting or ordering it.

Return a JSON like:
{{
    "Neurologist": ["MRI_Brain"],
    "Pulmonologist": ["Chest_X-ray"]
}}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return json.loads(response["choices"][0]["message"]["content"])


def compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
    answer = query_model(moderator_llm, "\nHere is the correct diagnosis: " + correct_diagnosis + "\n Here was the doctor dialogue: " + diagnosis + "\nAre these the same?", "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else.")
    return answer.lower()


def run_paired_mode(specialists, patient_agent, meas_agent, doctor_agent):
    dialogue = []

    medical_report = generate_doctor_report(doctor_agent, patient_agent, mode="medical_report")
    #dialogue.append(medical_report)

    if len(specialists) < 2:
        raise ValueError("At least 2 specialists are required for paired mode.")

    spec1, spec2 = specialists[0], specialists[1]

    specialist_objs = {
    spec1: SpecialistAgent(spec1, doctor_agent.scenario, medical_report=medical_report),
    spec2: SpecialistAgent(spec2, doctor_agent.scenario, medical_report=medical_report)
    }

    print(f"\n[Paired Specialist Mode] {spec1} <-> {spec2}")
    context = (
        #f"Doctor: {medical_report}\n"
        f"{spec1}: I have reviewed the case and have initial thoughts.\n"
        f"{spec2}: I'm looking forward to discussing the diagnosis.\n"
    )

    def run_pair_dialogue(spec1, spec2, prior_context, turns=6):
        convo = prior_context.strip() + "\n"
        current_speaker, other_speaker = spec1, spec2

        for i in range(turns):
            is_last_turn_for_speaker = (
                (turns % 2 == 0 and i >= turns - 2) or
                (turns % 2 == 1 and i == turns - 1)
            )

            response = specialist_objs[current_speaker].inference_specialist(
                convo, current_speaker, other_speaker
            )

            # Early stop if the speaker signals they’re done
            if any(phrase in response.lower() for phrase in ["i have no further", "this concludes", "we are aligned"]):
                print(f"{current_speaker} signaled conclusion. Ending early.")
                convo += f"{current_speaker}: {response.strip()}\n"
                break

            if is_last_turn_for_speaker:
                convo += f"{current_speaker}: {response.strip()}\n"
            else:
                # Strip Tentative Diagnosis from mid-conversation turns
                cleaned_response = re.sub(r"Tentative Diagnosis:.*", "", response, flags=re.IGNORECASE).strip()
                convo += f"{current_speaker}: {cleaned_response}\n"

            # Check for and insert test result if requested
            lower_resp = response.lower()
            if "request test" in lower_resp:
                match = re.search(r"request test\s*:\s*(.+)", lower_resp)
                if match:
                    test_name = match.group(1).strip()
                    test_query = f"REQUEST TEST: {test_name}"
                    test_result = meas_agent.inference_measurement(test_query)
                    print(f"Measurement: {test_result}")
                    convo += f"Moderator: {test_result.strip()}\n"

            current_speaker, other_speaker = other_speaker, current_speaker

        return convo

    paired_discussion = run_pair_dialogue(spec1, spec2, context)
    dialogue.append(f"Paired Discussion - {spec1} and {spec2}:\n{paired_discussion}")


    full_discussion = "\n\n".join(dialogue)
    # 🔻 Final diagnosis per specialist
    for spec in specialist_objs:
        final_prompt = f"""
You’ve now seen the full discussion and test results.
Please state your final diagnosis clearly.
Only include a 1-sentence justification and format your answer as:
Final Diagnosis: [Diagnosis] - [Justification]
"""
        response = query_model(
            specialist_objs[spec].backend,
            prompt=full_discussion + "\n\n" + final_prompt,
            system_prompt="You are a specialist completing your final diagnosis."
        )
        dialogue.append(f"{spec} Final Diagnosis: {response.strip()}")
    # Moderator short summary
    
    moderator_prompt = f"Review this discussion and provide a final diagnosis in under 100 characters:\n\n{full_discussion}\n\nModerator:"
    moderator_response = query_model(
        doctor_agent.backend,
        moderator_prompt,
        "You are a clinical moderator. Respond in under 100 characters."
    )
    dialogue.append(f"Moderator: {moderator_response}")
    return "\n".join(dialogue)


def run_scaffolding_mode(specialists, patient_agent, meas_agent, doctor_agent):
    dialogue = []


    medical_report = generate_doctor_report(doctor_agent, patient_agent, mode="medical_report")
    #dialogue.append(medical_report)


    specialist_objs = {name: SpecialistAgent(name, doctor_agent.scenario, medical_report=medical_report) for name in specialists}
    n = len(specialists)


    if n < 3 or n > 5:
        raise ValueError("Number of specialists must be between 3 and 5.")


    def run_pair_dialogue(spec1, spec2, prior_context, turns=6):
        convo = prior_context.strip() + "\n"
        current_speaker, other_speaker = spec1, spec2

        for i in range(turns):
            is_last_turn_for_speaker = (
                (turns % 2 == 0 and i >= turns - 2) or
                (turns % 2 == 1 and i == turns - 1)
            )

            response = specialist_objs[current_speaker].inference_specialist(
                convo, current_speaker, other_speaker
            )

            # Early stop if the speaker signals they’re done
            if any(phrase in response.lower() for phrase in ["i have no further", "this concludes", "we are aligned"]):
                print(f"{current_speaker} signaled conclusion. Ending early.")
                convo += f"{current_speaker}: {response.strip()}\n"
                break

            if is_last_turn_for_speaker:
                convo += f"{current_speaker}: {response.strip()}\n"
            else:
                # Strip Tentative Diagnosis from mid-conversation turns
                cleaned_response = re.sub(r"Tentative Diagnosis:.*", "", response, flags=re.IGNORECASE).strip()
                convo += f"{current_speaker}: {cleaned_response}\n"

            # Check for and insert test result if requested
            lower_resp = response.lower()
            if "request test" in lower_resp:
                match = re.search(r"request test\s*:\s*(.+)", lower_resp)
                if match:
                    test_name = match.group(1).strip()
                    test_query = f"REQUEST TEST: {test_name}"
                    test_result = meas_agent.inference_measurement(test_query)
                    print(f"Measurement: {test_result}")
                    convo += f"Moderator: {test_result.strip()}\n"

            current_speaker, other_speaker = other_speaker, current_speaker

        return convo

    # Stage 1: First two specialists discuss starting from doctor_report
    spec1, spec2 = specialists[0], specialists[1]
    context = (
        #f"Doctor: {medical_report}\n"
        f"{spec1}: I have reviewed the case and have initial thoughts.\n"
        f"{spec2}: I look forward to discussing the diagnosis.\n"
    )

    specialist_objs[spec1].medical_report = medical_report
    specialist_objs[spec2].medical_report = medical_report
    doctor_report_discussion = run_pair_dialogue(spec1, spec2, context)
    dialogue.append(f"Report 2 - {spec1} and {spec2} Discussion:\n{doctor_report_discussion}")


    # Stage 2: Remaining specialists build on prior specialist discussion only
    if n == 3:
        spec3 = specialists[2]
        context = f"{doctor_report_discussion}\n{spec3}, please review and add your reasoning."
        r3 = specialist_objs[spec3].inference_specialist(context, spec3, "Moderator")
        dialogue.append(f"Report 3 - {spec3} Review:\n{spec3}: {r3}\n")


    elif n == 4:
        spec3, spec4 = specialists[2], specialists[3]
        context = f"{doctor_report_discussion}\n{spec3} and {spec4}, discuss their assessments."
        report_3 = run_pair_dialogue(spec3, spec4, context)
        dialogue.append(f"Report 3 - {spec3} and {spec4} Discussion:\n{report_3}")


    elif n == 5:
        spec3, spec4, spec5 = specialists[2], specialists[3], specialists[4]
        context = f"{doctor_report_discussion}\n{spec3} and {spec4}, discuss their assessments."
        report_3 = run_pair_dialogue(spec3, spec4, context)
        dialogue.append(f"Report 3 - {spec3} and {spec4} Discussion:\n{report_3}")


        context_4 = f"{doctor_report_discussion}\n{report_3}\n{spec5}, synthesize prior reports and give final thoughts."
        r4 = specialist_objs[spec5].inference_specialist(context_4, spec5, "Moderator")
        dialogue.append(f"Report 4 - {spec5} Final Synthesis:\n{spec5}: {r4}\n")


    
        # Specialists provide final diagnosis
    full_discussion = "\n\n".join(dialogue)
    for spec in specialists:
        final_prompt = f"""
You’ve now seen the full discussion and test results.
Please state your final diagnosis clearly.
Only include a 1-sentence justification and format your answer as:
Final Diagnosis: [Diagnosis] - [Justification]
"""
        response = query_model(
            specialist_objs[spec].backend,
            prompt=full_discussion + "\n\n" + final_prompt,
            system_prompt="You are a specialist completing your final diagnosis."
        )
        dialogue.append(f"{spec} Final Diagnosis: {response.strip()}")

    # Moderator final synthesis
    moderator_prompt = f"Review all specialist reports and provide final diagnosis:\n\n{full_discussion}\n\nModerator:"
    moderator_response = query_model(
        doctor_agent.backend,
        moderator_prompt,
        "You are a senior clinical moderator synthesizing specialist reports. Respond in under 100 characters."
    )
    dialogue.append(f"Moderator: {moderator_response}")

    return "\n".join(dialogue)

#def run_group_mode(specialists, patient_agent, meas_agent, doctor_agent):
    dialogue = []

    doctor_report = generate_doctor_report(doctor_agent, patient_agent, mode="doctor_report")
    dialogue.append(doctor_report)

    if len(specialists) < 5:
        raise ValueError("Group mode requires at least 5 specialists.")

    specialist_objs = {name: SpecialistAgent(name, doctor_agent.scenario) for name in specialists}
    n = len(specialists)

    def run_pair_dialogue(spec1, spec2, prior_context, turns=4):
        convo = prior_context.strip() + "\n"
        current_speaker, other_speaker = spec1, spec2

        for i in range(turns):
            is_last_turn_for_speaker = (
                (turns % 2 == 0 and i >= turns - 2) or
                (turns % 2 == 1 and i == turns - 1)
            )

            response = specialist_objs[current_speaker].inference_specialist(
                convo, current_speaker, other_speaker
            )

            # Early stop if the speaker signals they’re done
            if any(phrase in response.lower() for phrase in ["i have no further", "this concludes", "we are aligned"]):
                print(f"{current_speaker} signaled conclusion. Ending early.")
                convo += f"{current_speaker}: {response.strip()}\n"
                break

            if is_last_turn_for_speaker:
                convo += f"{current_speaker}: {response.strip()}\n"
            else:
                # Strip Tentative Diagnosis from mid-conversation turns
                cleaned_response = re.sub(r"Tentative Diagnosis:.*", "", response, flags=re.IGNORECASE).strip()
                convo += f"{current_speaker}: {cleaned_response}\n"

            # Check for and insert test result if requested
            lower_resp = response.lower()
            if "request test" in lower_resp:
                match = re.search(r"request test\s*:\s*(.+)", lower_resp)
                if match:
                    test_name = match.group(1).strip()
                    test_query = f"REQUEST TEST: {test_name}"
                    test_result = meas_agent.inference_measurement(test_query)
                    print(f"Measurement: {test_result}")
                    convo += f"Moderator: {test_result.strip()}\n"

            current_speaker, other_speaker = other_speaker, current_speaker

        return convo

    # Stage 1
    spec1, spec2 = specialists[0], specialists[1]
    context = (
        f"Doctor: {doctor_report}\n"
        f"{spec1}: I have reviewed the case.\n"
        f"{spec2}: Let's discuss it thoroughly.\n"
    )
    report1 = run_pair_dialogue(spec1, spec2, context)
    dialogue.append(f"Group Report 1 - {spec1} and {spec2}:\n{report1}")

    # Stage 2
    spec3, spec4 = specialists[2], specialists[3]
    context2 = f"{report1}\n{spec3} and {spec4}, share your thoughts."
    report2 = run_pair_dialogue(spec3, spec4, context2)
    dialogue.append(f"Group Report 2 - {spec3} and {spec4}:\n{report2}")

    # Stage 3 (additional round for group mode)
    if n > 5:
        spec5, spec6 = specialists[4], specialists[5]
    else:
        spec5, spec6 = specialists[4], specialists[0]  # fallback

    context3 = f"{report1}\n{report2}\n{spec5} and {spec6}, provide additional insights."
    report3 = run_pair_dialogue(spec5, spec6, context3)
    dialogue.append(f"Group Report 3 - {spec5} and {spec6}:\n{report3}")

        # Specialists provide final diagnosis
    full_discussion = "\n\n".join(dialogue)
    for spec in specialists:
        final_prompt = f"""
You’ve now seen the full discussion and test results.
Please state your final diagnosis clearly.
Only include a 1-sentence justification and format your answer as:
Final Diagnosis: [Diagnosis] - [Justification]
"""
        response = query_model(
            specialist_objs[spec].backend,
            prompt=full_discussion + "\n\n" + final_prompt,
            system_prompt="You are a specialist completing your final diagnosis."
        )
        dialogue.append(f"{spec} Final Diagnosis: {response.strip()}")

    # Final Synthesis
    full_discussion = "\n\n".join(dialogue)
    moderator_prompt = f"Summarize the group specialist reports and provide a final diagnosis in under 100 characters:\n\n{full_discussion}\n\nModerator:"
    moderator_response = query_model(
        doctor_agent.backend,
        moderator_prompt,
        "You are a senior clinical moderator. Return a concise final diagnosis (max 100 characters)."
    )

    dialogue.append(f"Moderator: {moderator_response}")
    return "\n".join(dialogue)

def run_specialist_patient_interaction(specialist_name, patient_agent, meas_agent, scenario, total_turns=10):
    specialist = SpecialistAgent_AsDoctor(specialist_name, scenario)
    dialogue = ""
    
    for turn in range(total_turns):
        progress = int(((turn + 1) / total_turns) * 100)
        
        # Specialist asks question based on dialogue history
        question = specialist.inference_specialist(dialogue, specialist_name, "Patient")
        print(f"{specialist_name} [{progress}%]:\n{question.strip()}\n")
        
        # Check if specialist is ready to diagnose or end
        if any(phrase in question.lower() for phrase in ["diagnosis ready", "this concludes", "i have no further"]):
            print(f"{specialist_name} signals end of questioning.")
            dialogue += f"{specialist_name}: {question}\n"
            break
        
        # Use the SAME patient response as original doctor-patient conversation for this question
        # If you have stored the original patient answers, retrieve them here, e.g.:
        # patient_response = original_patient_response_for(question)
        # For now, just reuse patient_agent.inference_patient(question), but note this regenerates answer
        patient_response = patient_agent.inference_patient(question)
        print(f"Patient [{progress}%]: \n{patient_response.strip()}\n")
        
        dialogue += f"{specialist_name}: {question}\nPatient: {patient_response}\n"
        
        # Check if specialist requests a test
        if "request test" in question.lower():
            match = re.search(r"request test\s*:\s*(.+)", question.lower())
            if match:
                test_name = match.group(1).strip()
                test_query = f"REQUEST TEST: {test_name}"
                test_result = meas_agent.inference_measurement(test_query)
                print(f"Measurement [{progress}%]: {test_result}")
                dialogue += f"Measurement: {test_result}\n"
                patient_agent.add_hist(test_result)
        
        time.sleep(1)
    report_prompt = f"Based on the dialogue below, provide a concise one-sentence report summarizing your reasoning and findings.\n\n{dialogue}\n{specialist_name}:"
    final_report = query_model(specialist.backend, report_prompt, system_prompt="You are a specialist doctor providing a concise summary report.")

    diagnosis_prompt = f"Based on the dialogue below, provide your final diagnosis in one sentence.\n\n{dialogue}\nFinal Diagnosis:"
    final_diagnosis = query_model(specialist.backend, diagnosis_prompt, system_prompt="You are a specialist doctor providing your final diagnosis.")

    print(f"\n{specialist_name} Final Report: {final_report.strip()}")
    print(f"{specialist_name} Final Diagnosis: {final_diagnosis.strip()}\n")

    return dialogue, final_report.strip(), final_diagnosis.strip()
   
def aggregate_and_moderate(final_diagnoses, moderator_backend="gpt4"):
    from collections import Counter
    diagnosis_counts = Counter(final_diagnoses)
    majority_diagnosis = diagnosis_counts.most_common(1)[0][0]

    mod_prompt = (
        "Multiple specialists have provided the following final diagnoses:\n" +
        "\n".join(f"- {diag}" for diag in final_diagnoses) +
        f"\n\nPlease provide a final consensus diagnosis, agreeing with the majority diagnosis:\n{majority_diagnosis}\nModerator:"
    )

    moderator_summary = query_model(
        moderator_backend,
        mod_prompt,
        system_prompt="You are a clinical moderator synthesizing specialist reports into a final diagnosis."
    )
    print(f"Moderator: {moderator_summary.strip()}")
    return moderator_summary.strip()


def main(api_key, replicate_api_key, inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm, measurement_llm, moderator_llm, num_scenarios, dataset, img_request, total_inferences, anthropic_api_key=None):
    


    anthropic_llms = ["claude3.5sonnet"]
    replicate_llms = ["llama-3-70b-instruct", "llama-2-70b-chat", "mixtral-8x7b"]


    if patient_llm in replicate_llms or doctor_llm in replicate_llms:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    if doctor_llm in anthropic_llms:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key


    # Dataset loading
    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    else:
        raise Exception(f"Dataset {dataset} not supported")


    total_correct = 0
    total_presents = 0


    if "HF_" in moderator_llm:
        pipe = load_huggingface_model(moderator_llm.replace("HF_", ""))
    else:
        pipe = None


    if num_scenarios is None:
        num_scenarios = scenario_loader.num_scenarios

    def normalize_test_name(name):
        name = name.strip()
        name = re.sub(r'[\s\-]+', '_', name)
        name = "_".join(word.capitalize() for word in name.split('_'))
        return name
    
    for _scenario_id in range(0, min(num_scenarios, scenario_loader.num_scenarios)):
        total_presents += 1
        print(f"\n=== Scenario {_scenario_id} Starting ===")


        scenario = scenario_loader.get_scenario(id=_scenario_id)
        meas_agent = MeasurementAgent(scenario=scenario, backend_str=measurement_llm)
        patient_agent = PatientAgent(scenario=scenario, bias_present=patient_bias, backend_str=patient_llm)
        doctor_agent = DoctorAgent(scenario=scenario, bias_present=doctor_bias, backend_str=doctor_llm, max_infs=total_inferences, img_request=img_request)


        specialists = assign_specialists_and_explain_by_diagnosis(scenario)
        #print(f"Clean specialist names for dialogue: {specialists}")

        doctor_patient_dialogue = ""
        doctor_tests_requested = 0  # Make sure this is initialized before the loop

        print("\n[Regular Doctor-Patient Interaction]")
        max_tests_allowed = 5
        for _inf_id in range(total_inferences):
            doctor_question = doctor_agent.inference_doctor(doctor_patient_dialogue)
            progress = int(((_inf_id + 1) / total_inferences) * 100)
            print(f"Doctor [{progress}%]: {doctor_question}")

            
            if "REQUEST TEST" in doctor_question:
                try: 
                    # Check if doctor is within test request limit
                    if doctor_tests_requested < max_tests_allowed:
                        # Check if test_name is a valid vital sign from the scenario
                        vital_signs = scenario.physical_exams.get("Vital_Signs", {})
                        test_text = doctor_question.split("REQUEST TEST:")[-1].strip()
                        matched_test = None

                        # Check for full vital signs
                        if "vital signs" in test_text.lower():
                            matched_test = "Vital Signs"
                        else:
                            # Flexible matching for individual vitals
                            for vital_name in vital_signs:
                                if normalize_test_name(vital_name) in normalize_test_name(test_text):
                                    matched_test = vital_name
                                    break

                        if matched_test == "Vital Signs":
                            print(f"Doctor requested full set of Vital Signs.")
                            for vital_name, vital_value in vital_signs.items():
                                test_result = f"{vital_name}: {vital_value}"
                                print(f"Measurement [{progress}%]: {test_result}")
                                doctor_patient_dialogue += f"Doctor: {doctor_question}\nMeasurement: {test_result}\n"
                                patient_agent.add_hist(test_result)
                                meas_agent.add_hist(test_result)
                            doctor_tests_requested += 1

                        elif matched_test:
                            test_result = meas_agent.inference_measurement(f"REQUEST TEST: {matched_test.replace(' ', '_')}")
                            doctor_tests_requested += 1
                            print(f"Measurement [{progress}%]: {test_result}")
                            doctor_patient_dialogue += f"Doctor: {doctor_question}\nMeasurement: {test_result}\n"
                            patient_agent.add_hist(test_result)

                        else:
                            print(f"[Invalid Test] Doctor requested non-vital test: '{test_text}' — Ignored.")
                            doctor_patient_dialogue += f"Doctor: {doctor_question}\n[System]: Test '{test_text}' is not available.\n"
                            patient_agent.add_hist(f"Doctor attempted to request an unavailable test: {test_text}")
                            meas_agent.add_hist(f"Doctor attempted to request an unavailable test: {test_text}")

                    else:
                        print("Doctor has exceeded test request limit.")
                        doctor_patient_dialogue += f"Doctor: {doctor_question}\n[System]: Test request limit exceeded. No further tests allowed.\n"
                        patient_agent.add_hist("Doctor attempted a test but exceeded the maximum number of allowed test requests.")
                        meas_agent.add_hist("Doctor attempted a test but exceeded the maximum number of allowed test requests.")
                        continue
                except Exception as e:
                    print(f"[ERROR] Exception during test handling: {e}")
                    continue

                continue
            else:
                # Normal patient response
                patient_response = patient_agent.inference_patient(doctor_question)
                print(f"Patient [{progress}%]: {patient_response}")
                doctor_patient_dialogue += f"Doctor: {doctor_question}\nPatient: {patient_response}\n"
                meas_agent.add_hist(patient_response)

            time.sleep(1.0)

        # medical_report report to give specialists full objective case
        medical_report = generate_doctor_report(doctor_agent, patient_agent, mode="medical_report")


        # doctor_report report to simulate doctor bias or limited info
        medical_report_doctor_report = generate_doctor_report(doctor_agent, patient_agent, mode="doctor_report")

        #print("\n[Paired Mode Specialist Dialogue]")
        paired_dialogue = run_paired_mode(specialists, patient_agent, meas_agent, doctor_agent)
        print(paired_dialogue)

        print("\n[Scaffolding Mode Specialist Dialogue]")
        scaffolding_dialogue = run_scaffolding_mode(specialists, patient_agent, meas_agent, doctor_agent)
        print(scaffolding_dialogue)

        print("\n[Group: Patient-Specialist Dialogue]")

        dialogue = ""
        exam_info = scenario.exam_information()
        test_dict = exam_info.get("tests", {})
        available_tests = extract_test_names(test_dict)  
        available_images = extract_image_names(scenario)
        diagnosis = scenario.diagnosis_information()
        test_permissions = assign_tests_to_specialists(specialists, available_tests, scenario.diagnosis_information())
        image_permissions = assign_images_to_specialists(specialists, available_images, diagnosis)
        specialist_tests_requested = {specialist: 0 for specialist in specialists}
        
        all_final_reports = []
        all_final_diagnoses = []

        for specialist_name in specialists:
            allowed_tests = test_permissions.get(specialist_name, [])
            allowed_images = image_permissions.get(specialist_name, [])
            specialist = SpecialistAgent_AsDoctor(
                specialty=specialist_name,
                scenario=scenario,
                backend_str="gpt4",
                max_infs=total_inferences,
                allowed_images=allowed_images,
                allowed_tests=test_permissions.get(specialist_name, []),
                bias_present=doctor_bias,
                img_request=img_request
            )

            for turn in range(total_inferences):
                progress = int(((turn + 1) / total_inferences) * 100)

                question = specialist.inference_specialist(dialogue)
                print(f"{specialist_name} [{progress}%]: {question.strip()}")

                if any(phrase in question.lower() for phrase in ["diagnosis ready", "this concludes", "i have no further"]):
                    print(f"{specialist_name} signals end of questioning.")
                    dialogue += f"{specialist_name}: {question}\n"
                    break

                patient_response = patient_agent.inference_patient(question)
                print(f"Patient [{progress}%]: {patient_response.strip()}")

                dialogue += f"{specialist_name}: {question}\nPatient: {patient_response}\n"
                if "request test" in question.lower():
                    try: 
                        match = re.search(r"request test\s*:\s*(.+)", question.lower())
                        
                        
                        if match:
                            test_text = match.group(1).strip().replace('\n', '').replace('\r', '')
                            matched_test = None

                            for allowed_test in allowed_tests:
                                if normalize_test_name(allowed_test) in normalize_test_name(test_text):
                                    matched_test = allowed_test
                                    break

                            if matched_test:
                                if specialist_tests_requested[specialist_name] < max_tests_allowed:
                                    test_query = f"REQUEST TEST: {matched_test}"
                                    test_result = meas_agent.inference_measurement(test_query)
                                    specialist_tests_requested[specialist_name] += 1
                                    print(f"Measurement [{progress}%]: {test_result}")
                                    dialogue += f"Measurement: {test_result}\n"
                                    patient_agent.add_hist(test_result)
                                else:
                                    print(f"[LIMIT] {specialist_name} has exceeded their test request limit.")
                                    dialogue += f"{specialist_name}: REQUEST TEST: {matched_test}\n[System]: You have exceeded your test request limit.\n"
                                    patient_agent.add_hist(f"{specialist_name} attempted test '{matched_test}' but exceeded limit.")
                                    meas_agent.add_hist(f"{specialist_name} attempted test '{matched_test}' but exceeded limit.")
                            else:
                                print(f"[BLOCKED] {specialist_name} is not allowed to request test: {test_text}")
                                dialogue += f"{specialist_name}: REQUEST TEST: {test_text}\n[System]: You are not permitted to order this test.\n"
                                patient_agent.add_hist(f"{specialist_name} attempted unauthorized test '{test_text}'")
                                meas_agent.add_hist(f"{specialist_name} attempted unauthorized test '{test_text}'")

                        else:
                            print(f"[ERROR] Could not parse test request from specialist: {specialist_name}")
                            dialogue += f"{specialist_name}: {question}\n[System]: Could not understand the test request format.\n"
                            patient_agent.add_hist(f"{specialist_name} made an unclear test request.")
                            meas_agent.add_hist(f"{specialist_name} made an unclear test request.")
                            continue
                    except Exception as e:
                            print(f"[ERROR] Exception during test handling: {e}")
                            continue
                if "request image" in question.lower():
                    match = re.search(r"request image\s*:\s*(.+)", question.lower())
                    if match:
                        image_text = match.group(1).strip().replace('\n', '').replace('\r', '')
                        matched_image = None

                        for allowed_image in specialist.allowed_images:
                            if normalize_test_name(allowed_image) in normalize_test_name(image_text):
                                matched_image = allowed_image
                                break

                        if matched_image:
                            if specialist.images_requested < specialist.max_images_allowed:
                                specialist.images_requested += 1

                                raw_key = matched_image.replace(" ", "_")  # Match the JSON format
                                imaging_data = scenario.tests.get("Imaging", {}).get(raw_key, {})
                                findings = imaging_data.get("Findings", "No findings provided.")

                                print(f"[IMAGE] {specialist_name} requested image: {matched_image}")
                                dialogue += f"\n[{specialist_name} views {matched_image}]\nFindings: {findings}\n"
                                patient_agent.add_hist(f"{specialist_name} viewed image: {matched_image}")
                            else:
                                print(f"[LIMIT] {specialist_name} exceeded image request limit.")
                                dialogue += f"\n[System]: Image request limit reached.\n"
                        else:
                            print(f"[BLOCKED] {specialist_name} not permitted to request: {image_text}")
                            dialogue += f"\n[System]: You are not authorized to request image '{image_text}'.\n"
                    else:
                        print(f"[ERROR] Could not parse image request from specialist: {specialist_name}")
                        dialogue += f"{specialist_name}: {question}\n[System]: Could not understand the image request format.\n"
                        patient_agent.add_hist(f"{specialist_name} made an unclear image request.")

                time.sleep(1.0)
            #print(f"\n--- {specialist_name} Dialogue ---\n{dialogue}")
            print()

            report_prompt = (
                f"Based on the dialogue below, provide a concise one-sentence report summarizing your reasoning and findings.\n\n{dialogue}\n{specialist_name}:"
            )
            final_report = query_model(
                specialist.backend,
                report_prompt,
                system_prompt="You are a specialist doctor providing a concise summary report."
            )
            print(f"{specialist_name} Final Report: {final_report.strip()}")

            diagnosis_prompt = (
                f"Based on the dialogue below, provide your final diagnosis in one sentence.\n\n{dialogue}\nFinal Diagnosis:"
            )
            final_diagnosis = query_model(
                specialist.backend,
                diagnosis_prompt,
                system_prompt="You are a specialist doctor providing your final diagnosis."
            )
            print(f"{specialist_name} Final Diagnosis: {final_diagnosis.strip()}")
            print()
            all_final_reports.append(final_report.strip())
            all_final_diagnoses.append(final_diagnosis.strip())

        from collections import Counter
        diagnosis_counts = Counter(all_final_diagnoses)
        majority_diagnosis = diagnosis_counts.most_common(1)[0][0]

        moderator_prompt = (
            "Multiple specialists have provided the following final diagnoses:\n" +
            "\n".join(f"- {diag}" for diag in all_final_diagnoses) +
            f"\n\nPlease provide a final consensus diagnosis, agreeing with the majority diagnosis:\n{majority_diagnosis}\nModerator:"
        )

        moderator_response = query_model(
            doctor_agent.backend,
            moderator_prompt,
            system_prompt="You are a clinical moderator synthesizing specialist reports into a final diagnosis."
        )
        print(f"Moderator Final Consensus Diagnosis: {moderator_response.strip()}")
        print()
        print(f"Total tokens used - Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens}, Total: {total_tokens}")

       




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--openai_api_key', type=str, required=False, help='OpenAI API Key')
    parser.add_argument('--replicate_api_key', type=str, required=False, help='Replicate API Key')
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
    parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None', choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None', choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--doctor_llm', type=str, default='gpt4')
    parser.add_argument('--patient_llm', type=str, default='gpt4')
    parser.add_argument('--measurement_llm', type=str, default='gpt4')
    parser.add_argument('--moderator_llm', type=str, default='gpt4')
    parser.add_argument('--agent_dataset', type=str, default='MedQA') # MedQA, MIMICIV or NEJM
    parser.add_argument('--doctor_image_request', type=bool, default=False) # whether images must be requested or are provided
    parser.add_argument('--num_scenarios', type=int, default=None, required=False, help='Number of scenarios to simulate')
    parser.add_argument('--total_inferences', type=int, default=20, required=False, help='Number of inferences between patient and doctor')
    parser.add_argument('--anthropic_api_key', type=str, default=None, required=False, help='Anthropic API key for Claude 3.5 Sonnet')
   
    args = parser.parse_args()




    main(args.openai_api_key, args.replicate_api_key, args.inf_type, args.doctor_bias, args.patient_bias, args.doctor_llm, args.patient_llm, args.measurement_llm, args.moderator_llm, args.num_scenarios, args.agent_dataset, args.doctor_image_request, args.total_inferences, args.anthropic_api_key)


