import matplotlib.pyplot as plt
import numpy as np

# Example data
groups = ['African-American', 'Caucasian', 'Hispanic', 'Other', 'Native American', 'Asian']
mae_no_debiasing = [75.28, 62.77, 77.76, 78.43, 23.38, 97.54]
mae_with_debiasing = [71.81, 59.39, 74.95, 75.05, 14.81, 92.57]
mae_equalized = [68.69, 57.89, 73.46, 78.82, 10.99, 86.76]

fig, ax = plt.subplots()
bar_width = 0.25
index = np.arange(len(groups))

bar1 = ax.bar(index, mae_no_debiasing, bar_width, label='No Debiasing')
bar2 = ax.bar(index + bar_width, mae_with_debiasing, bar_width, label='With Debiasing')
bar3 = ax.bar(index + bar_width * 2, mae_equalized, bar_width, label='Equalized')

ax.set_xlabel('Racial Group')
ax.set_ylabel('Mean Absolute Error (Days)')
ax.set_title('Mean Absolute Error Across Groups for Date Predictions')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(groups)
ax.legend()

plt.show()
