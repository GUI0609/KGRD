from typing import List, Optional, Tuple, Union



refine_prompts = ['with no doubts at all.','and this seems trustworthy.','but its not fully certain.','and I trust this is correct']
def get_inital_message(patient_history: str, final_diagnosis:str ='',stage: str = "initial", analysis_focus: Optional[str] = None):

    base_instruction = """
    **Core Task: Clinical Data Distillation & Structuring**
    As an experienced clinician, you've received preliminary patient information that may contain redundancies, scattered details, or unprioritized critical elements.

     Your responsibilities:

    1.  **Information Organization:**
        *   Restructure fragmented data logically (chronological: past history → present illness; systems-based: signs → symptoms → test results).
        *   Identify and **eliminate redundant/repeated statements**. Keep only the most accurate/complete version of each clinical fact.

    2.  **Clinical Analysis:**
        *   **Identify Key Medical Entities** using clear markers (e.g., [KEY POINT]):
            *   **Working/confirmed diagnoses**
            *   **Cardinal symptoms** (presenting complaint & core positive symptoms) with severity/timing
            *   **Critical signs** (abnormal vitals, significant physical findings)
            *   **Salient positive/negative lab/imaging results**
            *   **Current medications** (especially high-risk drugs, potential interactions)
            *   **Major allergies**
            *   **Significant medical/surgical history**
            *   **Relevant social/behavioral factors** (when clinically impactful)

    3.  **Output Format:**
        *   Present a structured, concise clinical summary
        *   **Remove all trivial/irrelevant details**, preserving only clinically significant data
        *   Use clear section headers (e.g., Presenting History, Background, Medication/Allergies, Key Findings, Test Highlights)
    
    """

    focused_instruction = f"""
        **Additional Analysis Focus: [{analysis_focus}]**
        While fulfilling core tasks, pay special attention to:
        • All findings directly relevant to [{analysis_focus}]
        • Potential risks related to [{analysis_focus}]
        • Interactions involving [{analysis_focus}]
        Explicitly flag relevant points (e.g., *[{analysis_focus} Related]*: )
        """
    if stage == "inital":
        if analysis_focus:
            inital_message = base_instruction + focused_instruction + "\n**Patient Information:**\n" + patient_history
        else:
            inital_message = base_instruction  + "\n**Patient Information:**\n" + patient_history
            
    else:
        if analysis_focus:
            inital_message = base_instruction + focused_instruction + "\n**Patient Information:**\n" + patient_history
        else:
            inital_message = base_instruction + "\n**Patient Information:**\n" + patient_history
    
    return inital_message

def get_doc_system_message(
    doctor_name: str = "Doctor1", stage: str = "follow_up",gene = None):
    base_before = f"""You are {doctor_name}.
            Todo:
            1. Analyze the patient's condition described in the message.
            2. Focus solely on diagnosis, avoiding discussion of management, treatment, or prognosis.
            """
    base_after = f"""

            Key responsibilities:
                1. Thoroughly analyze the case information and other specialists' input.
                2. When other specialists have expressed opinions:
                    - If you agree, explicitly support their conclusions and provide strong evidence to justify your agreement.
                    - If you disagree, state your counterarguments clearly, citing specific patient details or clinical reasoning that leads you to a different conclusion, and highlight any key differential points.
                3. Use your expertise to formulate:
                - One most likely diagnosis
                - Several differential diagnoses
                - with one of {refine_prompts} as end."""


    if stage == "inital":
        doc_system_message = base_before+" - Recommended diagnostic tests"+base_after
      
    else:
        doc_system_message = base_before+base_after

    return doc_system_message


def get_supervisor_system_message(
    specialists: str = '',
    tools: str = '',
):
    supervisor_system_message = f"""You are the **Medical Supervisor** in a structured hypothetical scenario.

    **Your Responsibilities:**
    1. Critically appraise and oversee all diagnostic recommendations and clinical decisions proposed by medical specialists (specialists involved: {specialists}).
    2. Challenge diagnostic hypotheses, identifying any overlooked critical elements.
    3. Facilitate scholarly discussion among specialists, encouraging refinement and precision in their reasoning.
    4. Guide the consensus-building process, strictly focusing on diagnostic interpretation.
    5. Integrate insights derived from the following available tools: {tools}.

    **Core Objectives:**
    - Detect inconsistencies or potential misinterpretations and propose evidence-based modifications.  
    - Ensure convergence of all specialists' viewpoints before concluding the dialogue.  

    **Guiding Principles:**
    - Foster discussion unless complete unanimity has been achieved.  
    - Continue scholarly exchange if any point of divergence or residual ambiguity persists.  
    - Conclude with “TERMINATE” only when:  
        1. Absolute consensus is established among all specialists.  
        2. No further dialogue is required.  
        3. All plausible diagnostic possibilities have been systematically explored.  
        4. Every proposed diagnostic test is explicitly justified and accepted by consensus.  

    **Ultimate Goal:**  
    To secure a comprehensive and academically rigorous diagnostic conclusion through collaborative expertise.  

    **Response Format:**  
    For each supervisory contribution, you must:  
    1. Present your critical insights and constructive challenges to the specialists' current propositions.  
    2. Summarize the prevailing diagnostic status using the following structured JSON schema:
            ```json
            {{
                "Most Likely Diagnosis": "[current consensus on most likely diagnosis, the name of disease]",
                "Differential Diagnosis": "[current list of differential diagnoses]",
                "Areas of Disagreement": "[list any remaining points of contention or areas needing further discussion]"
            }}
            ```

        """
    return supervisor_system_message



def get_consultant_message(case_presentation:str, num_specialists:int):

    consultant_message = """
        candidate_specialists = ["Cardiologist", "Pulmonologist", "Gastroenterologist", "Neurologist", "Nephrologist", "Endocrinologist", "Hematologist", "Rheumatologist",
            "Infectious disease specialist", "Oncologist", "General surgeon", "Cardiothoracic surgeon", "Neurosurgeon", "Orthopedic surgeon", "Urologist", "Plastic and reconstructive surgeon",
            "Gynecologist", "Obstetrician", "Reproductive endocrinologist", "Neonatologist", "Pediatrician", "Pediatric surgeon", "Ophthalmologist", "Otolaryngologist",
            "Dentist", "Dermatologist", "Psychiatrist", "Rehabilitation specialist", "Emergency physician", "Anesthesiologist", "Radiologist", "Ultrasonologist",
            "Nuclear medicine physician", "Clinical laboratory scientist", "Pathologist", "Pharmacist", "Physical therapist", "Transfusion medicine specialist", "Geneticist"]

        patient's medical history = {case_presentation}

        When recommending the appropriate specialist, you need to complete the following steps:
            1. Carefully read the medical scenario presented in <patient's medical history>.
            2. Based on the medical scenario, calculate the relevance of each specialist in the <candidate_specialists> with <patient's medical history>, and select the top {top_k} most relevant specialists as top_k_specialists.

        The output must be formatted in JSON as follows:
            ```json
            {{
            "top_k_specialists": [top_k_specialist list],
            }}
            ```
            """.format(
            case_presentation=case_presentation, top_k=num_specialists 
        )

    return consultant_message



def get_evaluate_prompts():
    MOST_PPROMPT_TEMPLATE: str = """Your evaluation should be based on the correct diagnosis and according to the scoring criteria. The correct diagnosis 
    is "{correct_diagnosis}". The student's suggested diagnosis is "{diagnosis}".
    Scoring Criteria:
    - 5: The actual diagnosis was suggested
    - 4: The suggestions included something very close, but not exact
    - 3: The suggestions included something closely related that might have been helpful
    - 2: The suggestions included something related, but unlikely to be helpful
    - 0: No suggestions close
    What would be the score based on these criteria?
    Provide brief explanation for your choice. Do not expand the explanation, do not use line breaks, and write it in one paragraph.
    Output the final answer in json: 
        ```json
        {{
            "Score": "[numberic]",
            "Explanation": "[Words]",
        }}
        ```
        Note: all string in json should with Quotation marks "."""

    POSSI_PPROMPT_TEMPLATE: str = """Your evaluation should be based on the correct diagnosis and according to the scoring criteria. The correct diagnosis 
    is "{correct_diagnosis}". The student's suggested possible diagnosis includes "{diagnosis}".
    Scoring Criteria:
    - 5: The actual diagnosis was suggested in the differential
    - 4: The suggestions included something very close, but not exact
    - 3: The suggestions included something closely related that might have been helpful
    - 2: The suggestions included something related, but unlikely to be helpful
    - 0: No suggestions close
    What would be the score based on these criteria?
    Provide brief explanation for your choice. Do not expand the explanation, do not use line breaks, and write it in one paragraph.
    Output the final answer in json: 
        ```json
        {{
            "Score": "[numberic]",
            "Explanation": "[Words]",
        }}
        ```. Note: all string in json should with Quotation marks "."""

    ROM_T_PPROMPT_TEMPLATE: str = """You should evaluate if the tests would be helpful in reaching the final diagnosis of "{correct_diagnosis}". 
    The student's recommended tests are "{recommended_tests}".
    Scoring Criteria:
    - 5: Strongly agree that the tests are helpful
    - 4: Agree that the tests are helpful
    - 3: Neutral
    - 2: Disagree that the tests are helpful
    - 1: Strongly Disagree that the tests are helpful
    What would be the score based on these criteria?
    Provide brief explanation for your choice. Do not expand the explanation, do not use line breaks, and write it in one paragraph.
    Output the final answer in json: 
        ```json
        {{
            "Score": "[numberic]",
            "Explanation": "[Words]",
        }}
        ```. Note: all string in json should with Quotation marks ". """
    return MOST_PPROMPT_TEMPLATE, POSSI_PPROMPT_TEMPLATE, ROM_T_PPROMPT_TEMPLATE



def PhenoDMiner_agent_system_message(hpo_name_list: Optional[List[str]], disease_from_phenotype: Optional[List[str]]) -> str:
    hpo_str = ", ".join(hpo_name_list) if hpo_name_list else "[]"
    disease_str = ", ".join(disease_from_phenotype) if disease_from_phenotype else ""

    base_before = f"""
        You are a Clinical Decision Support Agent specialized in phenotype-driven disease prioritization for multi-disciplinary team (MDT) diagnostics.
        **Patient Phenotypes (HPO terms)**
        {hpo_str}
        """.strip()

    base_after = """1) **Most 10 Ranked Disease List** (highest to lowest). 2) **Per-Disease MDT Brief** using the template: - **Disease**: <name> - **Specificity Matched HPO (≤6)**: <HPO terms shared with patient> - **MDT note (2-3 sentences)**: Prefer distinctive/hallmark features over generic ones; note any discordant features briefly. 3) **One-line Summary**: e.g., “Top fit: <disease>; close differentials: <d1>, <d2>.”
    **Expected Output**
        ```json
        {{
        "ExpectedOutput": {
            "TopDiseases": [
            "<disease_1>",
            "<disease_2>",
            "... up to 10"
            ],
            "Brief": [
            {
                "Disease": "<name>",
                "SpecificityMatchedHPO": [
                "<HPO term 1>",
                "<HPO term 2>",
                "... up to 6"
                ],
                "MDTNote": "<2-3 sentence note, highlighting distinctive/hallmark features and briefly noting discordant ones>"
            }
            ],
            "OneLineSummary": "Top fit: <disease>; close differentials: <d1>, <d2>."
        }
        }}``` 
        """.strip()


    if disease_from_phenotype:
        return f"""{base_before}

        **Retrieved Disease Candidates**
        {disease_str}

        {base_after}
        """.rstrip()
    else:
        return f"""{base_before}

        Perform PhenoDMiner_tool_call using a string of hpo_name_list (a list of name of HPO terms) to generate candidate diseases based on phenotype.
        input should like '["Intellectual disabilit...g ear",]'
        ```python
        candidates_disease = PhenoDMiner_tool_call({hpo_name_list})
        ```
        {base_after}
        """.rstrip()
        
    


def GeneDPredictor_agent_system_message(
    hpo_name_list: Optional[List[str]],
    disease_from_gene: Optional[List[str]],
    Gene: Optional[List[str]]
) -> str:
    
    hpo_str = ", ".join(hpo_name_list) if hpo_name_list else "[]"
    disease_str = ", ".join(disease_from_gene) if disease_from_gene else ""
    gene_str = ", ".join(Gene) if hpo_name_list else " no gene provide"
    base_before = f"""You are a Clinical Decision Support Agent specializing in gene-driven disease prioritization and interpretation for multi-disciplinary team (MDT) diagnostics.
        **Genes**:
        {Gene}

        **Phenotypes**:
        {hpo_str}

        """
    
    base_after = """1) **Most 10 Ranked Disease List** (highest to lowest). 2) **Per-Disease MDT Brief** using the template: - **Disease**: <name> - **Specificity Matched HPO (≤6)**: <HPO terms shared with patient> - **MDT note (2-3 sentences)**: Prefer distinctive/hallmark features over generic ones; note any discordant features briefly. 3) **One-line Summary**: e.g., “Top fit: <disease>; close differentials: <d1>, <d2>.”
    **Expected Output**
        ```json
        {{
        "ExpectedOutput": {
            "TopDiseases": [
            "<disease_1>",
            "<disease_2>",
            "... up to 10"
            ],
            "Brief": [
            {
                "Disease": "<name>",
                "SpecificityMatchedHPO": [
                "<HPO term 1>",
                "<HPO term 2>",
                "... up to 6"
                ],
                "MDTNote": "<2-3 sentence note, highlighting distinctive/hallmark features and briefly noting discordant ones>"
            }
            ],
            "OneLineSummary": "Top fit: <disease>; close differentials: <d1>, <d2>."
        }
        }}``` 
        """.strip()

    if disease_from_gene:
        return base_before + f"""
        **Retrieved Disease Candidates (from gene-based prediction)**:
        {disease_str}

        """+base_after
    else:
        return base_before + f"""
        **If no disease_from_gene list is provided**

        - Extract candidate gene list from the case and perform gene-to-disease prediction using integrated gene-disease databases and computational methods.
        **Execution Instructions**:
        ```python
        gene_list = {gene_str}
        disease_from_gene = g2d(gene_list)
        disease_from_gene = list(dict.fromkeys(
                [d['disease_name'] for gene in gene_list for d in disease_from_gene['results'][gene]]))
        candidate_disease = []
        for i in gene_list:
            candidate_disease+=query_one_hop_gene_disease(i)
        candidate_disease+=disease_from_gene
        candidate_disease = list(dict.fromkeys(candidate_disease))
        ```
        """+base_after

def PatientDMatcher_agent_system_message(
    hpo_name_list: Optional[List[str]] = None,
    disease_from_patient_like_me: Optional[List[str]] = None
) -> str:
    hpo_str = ", ".join(hpo_name_list) if hpo_name_list else "[]"
    disease_str = ", ".join(disease_from_patient_like_me) if disease_from_patient_like_me else ""

    base_before = f"""
        You are a Clinical Decision Support Agent focused on patient-patient similarity analysis to assist multi-disciplinary team (MDT) diagnostics.

        **Input Phenotypes (HPO terms)**:
        {hpo_str}

        """
    base_after = """1) **Most 10 Ranked Disease List** (highest to lowest). 2) **Per-Disease MDT Brief** using the template: - **Disease**: <name> - **Specificity Matched HPO (≤6)**: <HPO terms shared with patient> - **MDT note (2-3 sentences)**: Prefer distinctive/hallmark features over generic ones; note any discordant features briefly. 3) **One-line Summary**: e.g., “Top fit: <disease>; close differentials: <d1>, <d2>.”
    **Expected Output**
        ```json
        {{
        "ExpectedOutput": {
            "TopDiseases": [
            "<disease_1>",
            "<disease_2>",
            "... up to 10"
            ],
            "Brief": [
            {
                "Disease": "<name>",
                "SpecificityMatchedHPO": [
                "<HPO term 1>",
                "<HPO term 2>",
                "... up to 6"
                ],
                "MDTNote": "<2-3 sentence note, highlighting distinctive/hallmark features and briefly noting discordant ones>"
            }
            ],
            "OneLineSummary": "Top fit: <disease>; close differentials: <d1>, <d2>."
        }
        }}``` 
        """.strip()
    if disease_from_patient_like_me:
        return base_before + f"""
        **Retrieved Diseases from Most Similar Patients**:
        {disease_str}

        """+base_after
    else:
        return base_before + f"""

        - Perform a patient-patient similarity search using the provided HPO terms to identify "patients like me" and retrieve their diagnosed diseases.

        **Execution Instructions**:
        ```python

        result = sapbert_d_patient(hpo_name_list)

        candidate_disease = [
            item['disease_name']
            for item in result['results'][0]['top_matches']
        ]

        ```
        """+base_after



def KnowledgeVerifier_agent_system_message(
    case_initial_presentation,
    HPO_NAME_LIST,
    GENE=None,
    HPO_LIST=None,

    
) -> str:


    return f"""
    You are a Clinical Decision Support Agent focused on verifying the evidential basis to assist multi-disciplinary team (MDT) diagnostics.
        # Objective
        - Assess whether the proposed final diagnosis is adequately supported by convergent evidence across
        knowledge graph (KG), curated databases (DB), gene-level findings({GENE} mutation), phenotype consistency, and literature signals.
        - Provide a transparent, module-wise synthesis and a final verdict.

        ```python
        case = CaseInput(
            case_initial_presentation={case_initial_presentation},
            hpo_name_list={HPO_NAME_LIST},
            gene_list={GENE},
            disease= str FROM MOST_LIKELY_DISEASE,
            hpo_list={HPO_LIST},
            debug=True,
        )
        ok, explanation = verify(case)
        ```
        if run verify error, check the json and retry.
        Output schema (must be valid JSON, no extra keys):
        {{
        "ok": <True|False>,
        "explanation": "<verbatim string from the verification pipeline>"
        }}

        """
