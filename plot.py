
# import json
# import matplotlib.pyplot as plt
# with open('logs\Learned files\Sequel Traning\Exp 8 - lr\Loss_resnet18_Ep_18_lr_0.00035_bs_12.json','r',encoding='utf-8') as f:
#     dct = json.load(f)
# list1=dct['Train Loss']
# plt.plot(list1)
# plt.show()

####################################################

import json
import matplotlib.pyplot as plt

file_paths = [ 'logs\Learned files\Sequel Experiment\graphs\Loss_resnet34_Ep_8_lr_0.00045_bs_18.json'
               

    
]

# Define different colors for each curve
colors = ['blue', 'orange', 'green', 'red', 'purple']

# Threshold value for highlighting the quickest-falling curve
threshold_loss = 0.5  # Set your threshold value here


quickest_falling_curve = None
quickest_falling_epochs = float('inf')  # Set a high initial value for comparison

for i, file_path in enumerate(file_paths[-5:]):  
    with open(file_path, 'r', encoding='utf-8') as f:
        dct = json.load(f)
        list1 = dct['Train_Accuracy']
        
        # Finding the epoch where the loss falls below the threshold
        for epoch, loss_value in enumerate(list1):
            if loss_value <= threshold_loss:
                if epoch < quickest_falling_epochs:
                    quickest_falling_curve = i
                    quickest_falling_epochs = epoch
                break
        
        # Set color for the quickest-falling curve
        color = colors[i] if i == quickest_falling_curve else 'gray'
        
        # Plot each curve with different colors and labeled accordingly
        plt.plot(list1, color=color, label=f'resNet34 {-(i - len(file_paths))}')

# Display legend and show plot
plt.legend()
plt.show()
