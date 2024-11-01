import matplotlib.pyplot as plt
import numpy as np

# Example data
groups = ['African-American', 'Caucasian', 'Hispanic', 'Other', 'Native American', 'Asian']
accuracy_no_debiasing = [73.6, 78.1, 75.27, 91.07, 66.67, 40.0]
accuracy_with_debiasing = [72.79, 74.49, 75.27, 76.79, 66.67, 20.0]

fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(groups))

bar1 = ax.bar(index, accuracy_no_debiasing, bar_width, label='No Debiasing')
bar2 = ax.bar(index + bar_width, accuracy_with_debiasing, bar_width, label='With Debiasing')

ax.set_xlabel('Racial Group')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Severity Prediction Accuracy Across Racial Groups')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(groups)

plt.show()

