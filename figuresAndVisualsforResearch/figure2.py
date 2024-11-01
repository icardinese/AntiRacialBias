import matplotlib.pyplot as plt
import numpy as np

groups = ['African-American', 'Caucasian', 'Hispanic', 'Other', 'Native American', 'Asian']

# Example data
fpr_no_debiasing = [15.47, 8.95, 7.78, 11.32, 0.0, 0.0]
fnr_no_debiasing = [22.16, 24.47, 28.28, 20.83, 0.0, 14.29]
fpr_with_debiasing = [16.65, 10.47, 8.38, 11.32, 0.0, 0.0]
fnr_with_debiasing = [20.42, 22.99, 27.27, 18.06, 0.0, 14.29]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(groups))

# Plot for False Positive Rate
bar1 = ax.bar(index, fpr_no_debiasing, bar_width, label='FPR No Debiasing', color='skyblue')
bar2 = ax.bar(index + bar_width, fpr_with_debiasing, bar_width, label='FPR With Debiasing', color='navy')

# Plot for False Negative Rate
bar3 = ax.bar(index + bar_width*2, fnr_no_debiasing, bar_width, label='FNR No Debiasing', color='lightcoral')
bar4 = ax.bar(index + bar_width*3, fnr_with_debiasing, bar_width, label='FNR With Debiasing', color='darkred')

ax.set_xlabel('Racial Group')
ax.set_ylabel('Rate (%)')
ax.set_title('False Positive and False Negative Rates Across Racial Groups')
ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(groups)
ax.legend()

plt.show()
