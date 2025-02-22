# Initially, libraries were imported to support different stages of the analysis.
import pandas as pd
from comet import download_model, load_from_checkpoint
import contractions
import language_tool_python


# The script loads translation data from a CSV file using pandas.
# The data is stored in a DataFrame (df) for easier manipulation.
# The CSV file is read with a specified delimiter and encoding.
# After loading the data, contractions in the columns “Original,” “BT1,” and “BT2” are expanded using the "contractions.fix() function".
# This preprocessing step ensures the text is normalized and ready for further analysis.
csv_path = r'file.csv'
df = pd.read_csv(csv_path, delimiter=";", encoding="utf-8")
df = pd.DataFrame(df)
df['Original'] = df['Original'].apply(lambda x: contractions.fix(str(x)))
df['BT1'] = df['BT1'].apply(lambda x: contractions.fix(str(x)))
df['BT2'] = df['BT2'].apply(lambda x: contractions.fix(str(x)))

# The relevant columns from the DataFrame are converted into Python lists for easier processing.
# These lists include the original sentences, translations from models T1 and T2, and back-translations BT1 and BT2.
Original = df["Original"].tolist()
T1 = df["T1"].tolist()
T2 = df["T2"].tolist()
T12 = df["T12"].tolist()
BT1 = df["BT1"].tolist()
BT2 = df["BT2"].tolist()

# To check grammar errors in translations, the script initializes LanguageTool for Brazilian Portuguese.After
# A loop iterates through each sentence in the T1 translations, checking for grammar issues.
# If issues are found, the script prints details about the error, including the error message, the problematic sentence, and a corrected version of the sentence using suggested replacements.
# If no errors are detected, the script simply states that the sentence is error-free.
# This same logic is applied to the T2 translations in a similar loop.

# LanguageTool grammar check
tool = language_tool_python.LanguageTool("pt-BR")
tool.language = "pt-BR"

line_number = 1
for line in df["T1"]: 
    matches = tool.check(line) 
    if matches:
        for match in matches:
            print(f"Line {line_number}: {match.message}")
            print(f"Error in sentence: '{match.sentence}'")
            start = match.offset
            end = match.offset + match.errorLength
            corrected_sentence = line[:start] + match.replacements[0] + line[end:]
            print(f"Corrected sentence: '{corrected_sentence}'")
    else:
        print(f"Line {line_number}: No errors found")
    line_number += 1
    
line_number = 1
for line in df["T2"]: 
    matches = tool.check(line) 
    if matches:
        for match in matches:
            print(f"Line {line_number}: ########################")
            print(f"Error message: {match.message}")
            print(f"Error in sentence: '{match.sentence}'")
            start = match.offset
            end = match.offset + match.errorLength
            corrected_sentence = line[:start] + match.replacements[0] + line[end:]
            print(f"Corrected sentence: '{corrected_sentence}'")
    else:
        print(f"Line {line_number}: No errors found")
    line_number += 1




# The script then downloads and loads the COMET model \texttt{Unbabel/XCOMET-XL}.
# This model is used for evaluating the quality of translations.
model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)


# Two functions are defined to handle translation evaluations.
# The first function evaluates translations without a reference translation, comparing only the source and machine translations.
# The second function includes a reference translation in the evaluation, enabling a more comprehensive assessment.

def evaluate_translations_no_reference(model, src_list, mt_list):
    data = [{"src": src, "mt": mt} for src, mt in zip(src_list, mt_list)]
    model_output = model.predict(data, batch_size=15)
    return model_output

def evaluate_translations_with_reference(model, src_list, mt_list, ref_list):
    data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(src_list, mt_list, ref_list)]
    model_output = model.predict(data, batch_size=15)
    return model_output

# A function is also defined to classify numerical COMET scores into descriptive categories, such as “weak,” “moderate,” “good,” “excellent,” and “optimal”.
# This provides a more interpretable way to understand translation quality.
def get_discrete_quality_score(score):
    if score <= 0.600:
        return 'weak'
    elif score <= 0.800:
        return 'moderate'
    elif score <= 0.940:
        return 'good'
    elif score <= 0.980:
        return 'excellent'
    else:
        return 'optimal'



# Using these functions, the script evaluates the quality of translations for T1, T2, T12, BT1, and BT2.
# For each translation, it calculates sentence-level and system-level scores.
# The system-level scores are then classified into qualitative categories using the classification function.
# The results are printed to provide detailed insights into the quality of each translation set.

# Evaluation of T1
Analysis_T1 = evaluate_translations_no_reference(model, Original, T1)
T1_sentence_scores = Analysis_T1.scores
T1_system_score = Analysis_T1.system_score
T1_classification = get_discrete_quality_score(T1_system_score)
print(f"System-level score: {T1_system_score:.3f}, Classification: {T1_classification}")

# Evaluation of T2
Analysis_T2 = evaluate_translations_no_reference(model, Original, T2)
T2_sentence_scores = Analysis_T2.scores
T2_system_score = Analysis_T2.system_score
T2_classification = get_discrete_quality_score(T2_system_score)
print(f"System-level score: {T2_system_score:.3f}, Classification: {T2_classification}")

# Evaluation of T12
Analysis_T12 = evaluate_translations_no_reference(model, Original, T12)
T12_sentence_scores = Analysis_T12.scores
T12_system_score = Analysis_T12.system_score
T12_classification = get_discrete_quality_score(T12_system_score)
print(f"System-level score: {T12_system_score:.3f}, Classification: {T12_classification}")

# Back Translation Evaluation
Analysis_BT1 = evaluate_translations_with_reference(model, T12, BT1, Original)
BT1_sentence_scores = Analysis_BT1.scores
BT1_system_score = Analysis_BT1.system_score
BT1_classification = get_discrete_quality_score(BT1_system_score)
print(f"System-level score: {BT1_system_score:.3f}, Classification: {BT1_classification}")

Analysis_BT2 = evaluate_translations_with_reference(model, T12, BT2, Original)
BT2_sentence_scores = Analysis_BT2.scores
BT2_system_score = Analysis_BT2.system_score
BT2_classification = get_discrete_quality_score(BT2_system_score)
print(f"System-level score: {BT2_system_score:.3f}, Classification: {BT2_classification}")

# Prepare DataFrame for Visualization
Comet_data = {
    'Items': list(range(1, 15)),
    'T1': [round(score, 3) for score in T1_sentence_scores],
    'T2': [round(score, 3) for score in T2_sentence_scores],
    'BT1': [round(score, 3) for score in BT1_sentence_scores],
    'BT2': [round(score, 3) for score in BT2_sentence_scores],
}
Comet_results = pd.DataFrame(Comet_data)

print("####################################### END #######################################")
