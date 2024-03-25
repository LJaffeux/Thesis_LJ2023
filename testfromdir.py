#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:33:06 2023

@author: jaffeux
"""
best_model=tuner.get_best_models(num_models=5)
j=0
for best in best_model:
    #best=best_model
    best.predict(img_generator)
    testfolder = os.getcwd()+'/All probes_v2/test'
    class_names=img_generator.class_names
    tab=[]
    i=0
    print('Model '+str(j))
    #retrieve class tags
    for folder in class_names: 
        
        for file in os.listdir(os.path.join(testfolder,folder)):
            tab.append(i)
        i=i+1
        
    img_generator2 = keras.preprocessing.image_dataset_from_directory(
                    os.path.join(testfolder), 
                    batch_size=32, 
                    image_size=(200,200),
                    color_mode="grayscale",
                    shuffle=False
                )
    preds=best.predict(img_generator2, verbose=1) 
    #predict classes from model
    predicted_classes=(np.argmax(np.round(preds),axis=1))
    c_mat=confusion_matrix(tab, predicted_classes)
    plt.figure(figsize=(8,6))
    #print confusion matrix
    niceconf=sns.heatmap(c_mat/np.sum(c_mat), xticklabels=class_names,
                         yticklabels=class_names,annot=True,
                fmt='.2%', cmap='Blues')
    
    plt.ylabel('Labelled as')
    plt.xlabel('Identified as')
    plt.savefig('niceconfALL_model'+str(j)+'.png',bbox_inches='tight')
    
    plt.clf()
    report=metrics.classification_report(tab, predicted_classes,
                                         target_names=class_names, 
                                         output_dict=True)
    dfreport=pd.DataFrame(report).transpose()
    categories=['CC',
                 'Co',
                 'CBC',
                 'CP',
                 'Dif',
                 'FA',
                 'HPC',
                 'SP',
                 'WD',
                'total acc.','macro avg.','weighted avg.']
    dfreport.index=categories
    #print report
    niceclassif=sns.heatmap(dfreport.drop('support',axis=1),
                            annot=True, 
                            linewidths=0.5, 
                            cmap='RdYlGn',
                            fmt=".2%",
                            vmin=0.6,
                            vmax=1)
    sns_plt=niceclassif.get_figure()
    sns_plt.savefig('nicereportALL_model'+str(j)+'.png')
    plt.close()
    j=j+1

# si c'est good best_model[0].save("modelCIP_1801.h5py")