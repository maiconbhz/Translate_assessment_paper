import pandas as pd
from comet import download_model, load_from_checkpoint
import contractions
import language_tool_python
import matplotlib.pyplot as plt
import numpy as np

# Colors
color1 = "#014040"
color2 = "#155252"
color3 = "#286464"
color4 = "#3C7676"
color5 = "#4F8989"

# CSV file path
csv_path = "file.csv"
df = pd.read_csv(csv_path, delimiter=";", encoding="utf-8")
df = pd.DataFrame(df)

# Fix contractions in specific columns
df['Original'] = df['Original'].apply(lambda x: contractions.fix(str(x)))
df['BT1'] = df['BT1'].apply(lambda x: contractions.fix(str(x)))
df['BT2'] = df['BT2'].apply(lambda x: contractions.fix(str(x)))

# Convert DataFrame columns to lists with English variable names
original_text = df["Original"].tolist()
translation1 = df["T1"].tolist()
translation2 = df["T2"].tolist()
translation12 = df["T12"].tolist()
back_translation1 = df["BT1"].tolist()
back_translation2 = df["BT2"].tolist()

########################## LanguageTool #########################################
# Initialize LanguageTool for English (change "en-US" if needed)
tool = language_tool_python.LanguageTool("pt-BR")
tool.language = "pt-BR"

# Check for errors in the first translation (T1)
line_num = 1
for line in df["T1"]:
    matches = tool.check(line)
    if matches:
        for match in matches:
            print(f"Line {line_num}: {match.message}")
            print(f"Error in sentence: '{match.sentence}'")
            start = match.offset
            end = match.offset + match.errorLength
            corrected_sentence = line[:start] + match.replacements[0] + line[end:]
            print(f"Corrected sentence: '{corrected_sentence}'")
    else:
        print(f"Line {line_num}: No error")
    line_num += 1

# Check for errors in the second translation (T2)
line_num = 1
for line in df["T2"]:
    matches = tool.check(line)
    if matches:
        for match in matches:
            print(f"Line {line_num}: ########################")
            print(f"Error message: {match.message}")
            print(f"Error in sentence: '{match.sentence}'")
            start = match.offset
            end = match.offset + match.errorLength
            corrected_sentence = line[:start] + match.replacements[0] + line[end:]
            print(f"Corrected sentence: '{corrected_sentence}'")
    else:
        print(f"Line {line_num}: No error")
    line_num += 1

###############################################################################
# Download and load the COMET model
model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)

def evaluate_translations_without_reference(model, source_list, mt_list):
    """
    Evaluate translations without a reference.
    """
    data = [{"src": src, "mt": mt} for src, mt in zip(source_list, mt_list)]
    model_output = model.predict(data, batch_size=15)
    return model_output

def evaluate_translations_with_reference(model, source_list, mt_list, reference_list):
    """
    Evaluate translations with a reference.
    """
    data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(source_list, mt_list, reference_list)]
    model_output = model.predict(data, batch_size=15)
    return model_output

def get_discrete_quality_score(score):
    """
    Convert a continuous score into a discrete quality rating.
    """
    if score <= 0.600:
        return 'weak'
    elif score <= 0.800:
        return 'moderate'
    elif score <= 0.940:
        return 'good'
    elif score <= 0.980:
        return 'excellent'
    else:
        return 'Optimal'
    
########################### Evaluation T1 #####################################
analysis_T1 = evaluate_translations_without_reference(model, original_text, translation1)
t1_sentence_scores = analysis_T1.scores
t1_system_score = analysis_T1.system_score
t1_error_spans = analysis_T1.metadata.error_spans
t1_classification = get_discrete_quality_score(t1_system_score)
print("Sentence-level scores:", [f"{score:.3f}" for score in t1_sentence_scores])
print(f"System-level score: {t1_system_score:.3f}")
print("Classification:", t1_classification)

mean_t1 = np.mean(t1_sentence_scores)
std_dev_t1 = np.std(t1_sentence_scores)
print(f"Mean: {mean_t1:.3f}")
print(f"Standard Deviation: {std_dev_t1:.3f}")

########################### Evaluation T2 #####################################
analysis_T2 = evaluate_translations_without_reference(model, original_text, translation2)
t2_sentence_scores = analysis_T2.scores
t2_system_score = analysis_T2.system_score
t2_error_spans = analysis_T2.metadata.error_spans
t2_classification = get_discrete_quality_score(t2_system_score)
print("Sentence-level scores:", [f"{score:.3f}" for score in t2_sentence_scores])
print(f"System-level score: {t2_system_score:.3f}")
print("Classification:", t2_classification)

mean_t2 = np.mean(t2_sentence_scores)
std_dev_t2 = np.std(t2_sentence_scores)
print(f"Mean: {mean_t2:.3f}")
print(f"Standard Deviation: {std_dev_t2:.3f}")

########################### Evaluation T12 #####################################
analysis_T12 = evaluate_translations_without_reference(model, original_text, translation12)
t12_sentence_scores = analysis_T12.scores
t12_system_score = analysis_T12.system_score
t12_error_spans = analysis_T12.metadata.error_spans
t12_classification = get_discrete_quality_score(t12_system_score)
print("Sentence-level scores:", [f"{score:.3f}" for score in t12_sentence_scores])
print(f"System-level score: {t12_system_score:.3f}")
print("Classification:", t12_classification)

mean_t12 = np.mean(t12_sentence_scores)
std_dev_t12 = np.std(t12_sentence_scores)
print(f"Mean: {mean_t12:.3f}")
print(f"Standard Deviation: {std_dev_t12:.3f}")

########################### BACKTRANSLATION #####################################
analysis_BT1 = evaluate_translations_with_reference(model, translation12, back_translation1, original_text)
bt1_sentence_scores = analysis_BT1.scores
bt1_system_score = analysis_BT1.system_score
bt1_error_spans = analysis_BT1.metadata.error_spans
bt1_classification = get_discrete_quality_score(bt1_system_score)
print("Sentence-level scores:", [f"{score:.3f}" for score in bt1_sentence_scores])
print(f"System-level score: {bt1_system_score:.3f}")
print("Classification:", bt1_classification)
print(bt1_sentence_scores)

mean_bt1 = np.mean(bt1_sentence_scores)
std_dev_bt1 = np.std(bt1_sentence_scores)
print(f"Mean: {mean_bt1:.3f}")
print(f"Standard Deviation: {std_dev_bt1:.3f}")

########################### BACKTRANSLATION #####################################
analysis_BT2 = evaluate_translations_with_reference(model, translation12, back_translation2, original_text)
bt2_sentence_scores = analysis_BT2.scores
bt2_system_score = analysis_BT2.system_score
bt2_error_spans = analysis_BT2.metadata.error_spans
bt2_classification = get_discrete_quality_score(bt2_system_score)
print("Sentence-level scores:", [f"{score:.3f}" for score in bt2_sentence_scores])
print(f"System-level score: {bt2_system_score:.3f}")
print("Classification:", bt2_classification)

mean_bt2 = np.mean(bt2_sentence_scores)
std_dev_bt2 = np.std(bt2_sentence_scores)
print(f"Mean: {mean_bt2:.3f}")
print(f"Standard Deviation: {std_dev_bt2:.3f}")

########################### Add BACKTRANSLATION DataFrame Columns #####################################
df["T12_x_BT1_x_Original"] = pd.Series(bt1_sentence_scores).round(3)
df["BT1_Classification"] = df["T12_x_BT1_x_Original"].apply(get_discrete_quality_score)
df["T12_x_BT2_x_Original"] = pd.Series(bt2_sentence_scores).round(3)
df["BT2_Classification"] = df["T12_x_BT2_x_Original"].apply(get_discrete_quality_score)
 
# Format values to three decimal places in the dictionary
comet_data = {
    'Item': list(range(1, len(translation1) + 1)),
    'T1': [round(score, 3) for score in t1_sentence_scores],
    'T2': [round(score, 3) for score in t2_sentence_scores],
    'BT1': [round(score, 3) for score in bt1_sentence_scores],
    'BT2': [round(score, 3) for score in bt2_sentence_scores],
}

comet_results = pd.DataFrame(comet_data)

# Plot COMET scores for translations (T1 and T2)
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35  # Width of the bars
index = np.arange(len(comet_results['Item']))  # X-axis positions

bar1 = ax.bar(index, comet_results['T1'], bar_width, label='Translation (T1)', color=color1)
bar2 = ax.bar(index + bar_width, comet_results['T2'], bar_width, label='Translation (T2)', color=color5)

ax.set_xlabel('Items', fontsize=12)
ax.set_ylabel('COMET Score (a.u.)', fontsize=12)
ax.set_title('', fontsize=14, pad=20)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(comet_results['Item'], fontsize=10)
ax.legend(fontsize=12, loc='lower left')
ax.axhline(y=0.980, color=color1, linestyle='--', linewidth=1, label='Reference (0.980)')

for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=5)

plt.ylim(0, 1.1)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot COMET scores for back translations (BT1 and BT2)
fig, ax = plt.subplots(figsize=(12, 6))
bar1 = ax.bar(index, comet_results['BT1'], bar_width, label='Back Translation (BT1)', color=color1)
bar2 = ax.bar(index + bar_width, comet_results['BT2'], bar_width, label='Back Translation (BT2)', color=color5)

ax.set_xlabel('Items', fontsize=12)
ax.set_ylabel('COMET Score (a.u.)', fontsize=12)
ax.set_title('', fontsize=14, pad=20)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(comet_results['Item'], fontsize=10)
ax.legend(fontsize=12, loc='lower left')
ax.axhline(y=0.980, color=color1, linestyle='--', linewidth=1, label='Reference (0.980)')

for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=5)

plt.ylim(0, 1.1)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("####################################### END #######################################")

