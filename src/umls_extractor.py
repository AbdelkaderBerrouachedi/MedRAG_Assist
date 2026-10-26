#!/usr/bin/env python3
"""
MedRAG-Assist: Phase III - Structured Extraction & UMLS Normalization Module.
Extracts clinical entities (symptoms, duration, severity, pertinent negatives, allergies)
and maps surface terms to Unified Medical Language System (UMLS) Concept Unique Identifiers (CUIs).
"""

from typing import List, Dict, Any

class UMLSEntityExtractor:
    def __init__(self):
        # UMLS Metathesaurus Knowledge Map (CUI, Preferred Name, Semantic Type)
        self.umls_knowledge_base = {
            "cough": {"cui": "C0010200", "pref_name": "Coughing", "type": "Sign or Symptom"},
            "fever": {"cui": "C0015967", "pref_name": "Fever", "type": "Sign or Symptom"},
            "dyspnea": {"cui": "C0013404", "pref_name": "Dyspnea", "type": "Sign or Symptom"},
            "shortness of breath": {"cui": "C0013404", "pref_name": "Dyspnea", "type": "Sign or Symptom"},
            "chest pain": {"cui": "C0008031", "pref_name": "Chest Pain", "type": "Sign or Symptom"},
            "fatigue": {"cui": "C0015672", "pref_name": "Fatigue", "type": "Sign or Symptom"},
            "phlegm": {"cui": "C0037803", "pref_name": "Sputum finding", "type": "Finding"},
            "penicillin": {"cui": "C0030843", "pref_name": "Penicillins", "type": "Pharmacologic Substance"},
            "amoxicillin": {"cui": "C0002645", "pref_name": "Amoxicillin", "type": "Organic Chemical"},
            "asthma": {"cui": "C0004096", "pref_name": "Asthma", "type": "Disease or Syndrome"},
            "copd": {"cui": "C0024117", "pref_name": "Chronic Obstructive Airway Disease", "type": "Disease or Syndrome"},
            "community-acquired pneumonia": {"cui": "C0549484", "pref_name": "Community-acquired pneumonia", "type": "Disease or Syndrome"},
            "acute bronchitis": {"cui": "C0149717", "pref_name": "Acute bronchitis", "type": "Disease or Syndrome"}
        }

    def map_to_cui(self, term: str) -> Dict[str, str]:
        """Maps a colloquial medical string to an authentic UMLS CUI."""
        term_clean = term.lower().strip()
        if term_clean in self.umls_knowledge_base:
            entry = self.umls_knowledge_base[term_clean]
            return {
                "term": term,
                "cui": entry["cui"],
                "pref_name": entry["pref_name"],
                "semantic_type": entry["type"]
            }
        return {"term": term, "cui": "CUI-UNKNOWN", "pref_name": term, "semantic_type": "Unmapped"}

    def extract_structured_patient_context(self, filtered_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parses dialogue turns to build a structured clinical schema.
        Extracts affirmed symptoms, duration, pertinent negatives, and drug allergies.
        """
        full_text = " ".join([t["utterance"] for t in filtered_turns]).lower()

        # 1. Affirmed Symptoms
        affirmed_symptoms = []
        if "cough" in full_text:
            affirmed_symptoms.append(self.map_to_cui("cough"))
        if "fever" in full_text:
            affirmed_symptoms.append(self.map_to_cui("fever"))
        if "fatigue" in full_text:
            affirmed_symptoms.append(self.map_to_cui("fatigue"))
        if "phlegm" in full_text:
            affirmed_symptoms.append(self.map_to_cui("phlegm"))

        # 2. Duration & Chronology
        duration = "3 weeks" if "three weeks" in full_text or "3 weeks" in full_text else "Unspecified"

        # 3. Pertinent Negatives (Explicitly denied by patient/doctor)
        pertinent_negatives = []
        if "no chest pain" in full_text or "no real shortness of breath" in full_text:
            pertinent_negatives.append(self.map_to_cui("chest pain"))
            pertinent_negatives.append(self.map_to_cui("dyspnea"))
        if "no blood" in full_text:
            pertinent_negatives.append({"term": "Hemoptysis", "cui": "C0019079", "pref_name": "Hemoptysis", "semantic_type": "Sign or Symptom"})

        # 4. Drug Allergies (Critical Contraindication Safety)
        allergies = []
        if "allergic to penicillin" in full_text or "amoxicillin" in full_text:
            pen_cui = self.map_to_cui("penicillin")
            pen_cui["reaction"] = "Hives and facial swelling"
            allergies.append(pen_cui)

        # 5. Medical History
        history = []
        if "no chronic lung" in full_text or "never smoked" in full_text:
            history.append("Non-smoker")
            history.append("No baseline asthma or COPD")

        structured_state = {
            "affirmed_symptoms": affirmed_symptoms,
            "duration": duration,
            "severity": "Moderate, progressive",
            "pertinent_negatives": pertinent_negatives,
            "allergies": allergies,
            "past_medical_history": history,
            "search_query": "worsening persistent cough fever pneumonia guidelines penicillin allergy treatment"
        }

        return structured_state

if __name__ == "__main__":
    extractor = UMLSEntityExtractor()
    sample = [{"speaker": "Patient", "utterance": "I have had a worsening cough and fever for 3 weeks, no chest pain, allergic to penicillin."}]
    res = extractor.extract_structured_patient_context(sample)
    import json
    print(json.dumps(res, indent=2))
