import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "MineChat"
#print("Let's chat! (type 'quit' to exit)")
sentences = {"Hello" : "greeting",
             "Hi" : "greeting",
             "What is the criteria of employement?" : "employedcriteria", 
             "What is Mine?" : "Minedef1952", 
             "What is the function of a inspector?" : "function_of_inspector_1952",
             "What powers does the Chief Inspector have to authorizing other Inspectors?": "chief_inspector_authorization_func_1952",
             "Who are considered public servants in The Coal act" : "chief_inspector_public_servant_1952",
             "What are Limitation of district magistrate's power in mining" : "district_magistrate_powers_sm_1952",
             "What are the rules related to Indian Standard Time" : "indian_standard_time_rules_1952",
             "What is working above ground?" : "working_above_ground_def1952",
             "What is relay and shift?" : "relay_and_shift_definition_def1952",
             "Bye Bye, Bro": "goodbye",
             "What is your purpose" : "items",
             "What is mine act 1952" : "Minesact1952def",
             "Where is the coal act 1952 applicable?" : "extent1952",
             "In which place is the coal act 1952 not applicable?" : "Kashmir1952",
             "Who is adult?" : "adult1952",
             "Who is known as an adult?" : "adult1952",
             "Who is agent in coal act 1952?" : "agent1952",
             "Who is the chief inspector?" : "cheifinspectordef1952",
             "What is day" : "daydef1952",
             "Who is district magistrate?" : "Districtmagistratedef",
             "Who is employed in a mine?" : "employeddef",
             "What is the criteria to be employed" : "employedcriteria",
             "Who is not employed" : "employeeexcluded",
             "Define inspector" : "inspectordef1952",
             "What are criteria of a mine?" : "Minescriteria1952",
             "Define Mines office." : "officeofmines1952",
             "What is open cast working" : "open_cast_workingdef1952",
             "Who is owner of the mines" : "mine_owner_definitiondef1952",
             "explain 'prescribed'" : "definition_prescribeddef1952",
             " Explain 'qualified medical practitioner'" : "qualified_medical_practitioner_definition_def1952",
             "What are 'rules' and 'regulations in the coal act?" : "regulations_rules_bye-laws_definition_def1952",
             "Define 'relay'" : "relay_and_shift_definition_def1952",
             "What do you mean by 'reportable injury'?" : "reportable_injury_definition_def1952",
             "Define 'serious bodily injury'" : "serious_bodily_injury_definition_def1952",
             " Tell me what is week" : "week_definition_def1952",
             "Define 'working below ground'" : "working_below_ground_def1952",
             "Define 'working above ground'" : "working_above_ground_def1952",
             "explain when the Coal Act 1952 does not apply?" : "act_not_applying_certain_cases_1952",
             "What is time of day in the coal act" : "coal_act_time_references_def_1952",
             "clarify the Indian standard time rules?" : "indian_standard_time_rules_1952",
             "What are the requirements for applying local mean time" : "indian_standard_time_rules_1952",
             "How is inspector appointed?" : "coal_act_inspector_appointment_1952",
             "Who has the authority to appoint inspectors of mines?" : "coal_act_inspector_appointment_1952",
             "How is someone eligible to be Cheif Inspector of mine" : "chief_inspector_restrictions_1952",
             "eligibility criteria for Chief Inspector and Inspectors" : "chief_inspector_restrictions_1952",
             "limitations on the district magistrate's powers as an Inspector?" : "district_magistrate_powers_long_1952",
             "Explain district magistrate's role as Inspector" : "district_magistrate_powers_sm_1952",
             "Who can be considered public servants in the coal act?" : "chief_inspector_public_servant_1952",
             "What does an inspector do?" : "function_of_inspector_1952",
             "Under this Act What power do Chief Inspectors and Inspectors have?" : "inspector_powers",
             "What is The Coal Act 1952 section-8?" : "government_person_authorization_1952_sec8",
             "Who are eligible to enter a mine for leveling and measuring?" : "government_person_authorization_1952_sec8",
             "Tell me the conditions for Government personnel to enter a mine?" : "government_person_authorization_1952_sec8",
             "What is The Coal Act 1952 section-9?" : "facilities_for_entry_inspection",
             "What are the Health survey in mines?" : "safety_inspection_1952_sec9a",
             "Can a mine fail safety survey?" : "safety_inspection_1952_sec9a",
             "What is The Coal Act 1952 Section-9A Sub-section-2?" : "examination_in_mine_1952_sec9a",
             "Tell me about safety measures in mines" : "examination_in_mine_1952_sec9a",
             "Tell me the type of information collected during a mine survey?" : "examination_in_mine_1952_sec9a",
             "What is The Coal Act 1952 Section-9A Sub-section-3?" : "working_time_in_mine_survey_1952_sec9a",
             "What are the regulations governing overtime pay in mines?" : "overtime_calculation_section_33_subsection_1",
             "What is the compensation for overtime during surveys?" : "working_time_in_mine_survey_1952_sec9a",
             "definition of the 'ordinary rate of wages'." : "ordinary_rate_of_wages_1952_sec9a",
             "What are the regulations regarding the 'ordinary rate of wages' in mines?" : "ordinary_rate_of_wages_1952_sec9a",
             "What do you mean by the Coal Act 1952 Section-9A Sub-section-4." : "medical_treatment_for_unfit_workers_1952_sec9a",
             "What is medical treatment of workers?" : "medical_treatment_for_unfit_workers_1952_sec9a",
             "What is the any legal framework governing medical treatment for unfit mine workers?" : "medical_treatment_for_unfit_workers_1952_sec9a",
             "What is the Coal Act 1952 Section-9A Sub-section-5." : "alternative_employment_for_medically_unfit_workers_1952_sec9a",
             "How can the disability allowance rate determined?" : "alternative_employment_for_medically_unfit_workers_1952_sec9a",
             "What is the Coal Act 1952 Section-9A Sub-section-6." : "determination_of_rates_for_disabilities_1952_sec9a",
             "Who can determining the rates for disabilities?" : "determination_of_rates_for_disabilities_1952_sec9a",
             "Tell me the Coal Act 1952 section-10 sub-section-1?" : "confidentiality_of_mine_records_1952_sec9a",
             "What is the process for determining when disclosure is necessary?" : "section_67_the_mines_act_employment_contravention",
             "is it viable for a mine workers access their own inspection records?" : "confidentiality_of_mine_records_1952_sec9a",
             "Tell me the Coal Act 1952 section-10 sub-section-2" : "exceptions_to_mine_information_disclosure_1952_sec9a",
             "Who can disclose mine information?" : "exceptions_to_mine_information_disclosure_1952_sec9a",
             "What are the legal regulations regarding exceptions to mine information disclosure?" : "exceptions_to_mine_information_disclosure_1952_sec9a",
             "What is the Coal Act 1952 section-10 sub-section-3" : "punishment_for_information_disclosure_1952_sec9a",
             "can fines be imposed for unauthorized disclosure?": "punishment_for_information_disclosure_1952_sec9a",
             "What is the penalty for unauthorized disclosure of mine information?" : "punishment_for_information_disclosure_1952_sec9a",
             "What is the Coal Act 1952 section-10 sub-section-4" : "sanction_for_trial_under_coal_act_1952_sec9a",
             "What is the Coal Act 1952 section-11 sub-section-1" : "appointment_of_certifying_surgeons_1952_sec9a",
             "explain the responsibilities of certifying surgeons?" : "appointment_of_certifying_surgeons_1952_sec9a",
             "What is the limitations on the number of certifying surgeons that can be appointed?" : "appointment_of_certifying_surgeons_1952_sec9a",
             "Can there be any mechanism for appealing authorization decisions?" : "authorization_of_medical_practitioners_by_certifying_surgeon_1952_sec9a",
             "What is the process for referring accidents or matters to the Committee?" : "duties_and_functions_of_committee_under_coal_act_1952_sec13",
             "What is the Coal Act 1952 section-13 sub-section-1" : "duties_and_functions_of_committee_under_coal_act_1952_sec13",
             "What is the timeline for the Committee to complete its tasks?" : "duties_and_functions_of_committee_under_coal_act_1952_sec13",
             "What do you mean by the Coal Act 1952 section-13 sub-section-2" : "chief_inspector_role_in_committee_proceedings_coal_act_1952_sec13",
             "Can individuals appeal to the Chief Inspector?" : "chief_inspector_role_in_committee_proceedings_coal_act_1952_sec13",
             "What is the Coal Act 1952 section-14 sub-section-1?" : "committee_powers_under_section_12_coal_act_1952_sec14",
             "What is the process for a Committee to request Inspector powers?" : "committee_powers_under_section_12_coal_act_1952_sec14",
             "What is the Coal Act 1952 section-14 sub-section-2" : "committee_powers_under_section_14_coal_act_1952",
             "are there any limitations or restrictions on these powers?" : "committee_powers_under_section_14_coal_act_1952",
             "What is the Coal Act 1952 section-15" : "expense_recovery_for_committee_inquiry_under_section_15",
             "What is the time limit for the owner or agent to make the payment?" : "expense_recovery_for_committee_inquiry_under_section_15",
             "Who Can give the notice?" : "notice_before_commencement_of_mining_operation_under_section_16",
             "When Should the notice be given" : "notice_timing_under_section_16",
             "What is the Coal Act 1952 section-17 sub-section-1" : "manager_qualification_and_appointment_under_section_17",
             "Are these responsibilities regularly monitored and enforced?" : "responsibility_for_matters_under_rules_in_coal_act_1952_subsection_2_section_18"
             }






total = 0
right = 0
for sentence in sentences:
    # sentence = "do you use credit cards?"
    #sentence = sentences[i]
    cp = sentence
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        if tag == sentences[cp]:
            right = right + 1
        else:
            print(f"{cp}")
        if cp == "I do not understand...":
            print(f"####{cp}####")
    total = total + 1
percentage = float((right * 100)/total)
print(f"Model Accuracy: {percentage} %")
#print(f"Total Cases: {total}\nAccurate Cases: {right}")