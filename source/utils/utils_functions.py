import source.fairness as fm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
import random
from math import pi
import plotly.graph_objects as go


def estimatingFM(sub_sample_size, num_rep, nan_treatment, fm_function, **kwargs):
    '''
    It returns the estimated fairness metric defined in fm_function by using bootstrapping method
    
    Input:
    sub_sample_size: number (can be int or float), provides the size of the sub-sample for the estimation trhough bootstrapping
                                                if it is int, it is used directly for the sub-sample size.
                                                if it is float, the size is computed as int(number_samples*sub_sample_size)
    num_rep: int, number of repetitions for bootstrapping procedure
    nan_treatment: boolean, if true replace nan to 0, otherwise, not consider them for computation.
    fm_function: function from source.fairness class, function to compute the fairness metrics.
    **kwargs: args, set of arguments related to fm_function in order to compute the fairness metric.
    
    Output:
    mean: float, provides the mean estimation
    std: float, provides the standard deviation estimation
    metrics: numpy (num_rep,), provides the array with the computation of the fairness metrics using bootstrapping.
    '''
    
    metrics = []
    args = {}
    
    for i in range(num_rep):
        #Drawing subsample
        if isinstance(sub_sample_size, int):
            selected = random.choices(range(kwargs['protected_attr'].shape[0]), k=sub_sample_size)
        else:
            selected = random.choices(range(kwargs['protected_attr'].shape[0]), k=int(kwargs['protected_attr'].shape[0]*sub_sample_size))
        
        args = kwargs.copy()
        for key, value in args.items():
            if key in ('y_true', 'y_pred', 'protected_attr'):
                args[key] = value[selected]
        metric = fm_function(**args)
        metrics.append(abs(metric))
    
    if nan_treatment:
        mean = np.mean(np.nan_to_num(metrics))
        std = np.std(np.nan_to_num(metrics))
        metrics = np.nan_to_num(metrics)
    else:
        metrics = np.array(metrics)
        mean = np.mean(metrics[np.isnan(metrics)==False])
        std = np.std(metrics[np.isnan(metrics)==False])
    
    return mean, std, metrics

def plotModelSingle(y_pred, y_true, prot_att, risks, positive_class, priv, bins, alpha, beta, sample_size, num_rep, nan_treatment=False, ascending=False, accumulative=True, bart = False): 
    '''
    Print the overall abs(demP), abs(eqOPP), and abs(eqODD) and a plot of these metric for each bins,
    where the bins and composed by individuals sorted by the risk level in descending order.
    
    Inputs:
    y_pred: numpy (n,1), contains the prediciton of the model
    y_true: numpy (n,1), contains the ground truth
    prot_att: numpy (n,1), contains the value for protected attribute
    risks: numpy (n,1), conatains the risks
    priv: object, provides the priviliged class. The type should be consistent with the values in prot_att
    positive_class: obejct, provides the positive class in y_true. The type should be consisten with the values in y_true and y_pred
    bins: int, number of bins to create
    alpha: float, provides the value used for computing risks
    beta: float, provides the value used for computing beta
    sample_size: number (can be int or float), provides the size of the sub-sample for the estimation trhough bootstrapping
    num_rep: int, number of repetitions for bootstrapping procedure
    nan_treatment: boolean, if true replace nan to 0, otherwise, not consider them for computation.
    ascending: boolean, if true, the bins in the plots are sorted in an ascending order, otherwise, they are sorted in a descending order
    accumulaive: boolean, if true, the fairness metrics are computed in a accumulative way, otherwise, each fairness metric is computed 
                          by using only samples in each bin
    
    Outputs:
    demP: list (len: bins), returns the demP estimated for each bin
    eqOPP: list (len: bins), returns the eqOPP estimated for each bin
    eqODD: list (len: bins), returns the eqODD estimated for each bin
    demP_ci: numpy (bins,2), returns the computed confidence interval of demP for each bin
    eqOPP_ci: numpy (bins,2), returns the computed confidence interval of eqOPP for each bin
    eqODD_ci: numpy (bins,2), returns the computed confidence interval of eqODD for each bin
    demmetrics: numpy (bins, num_rep), returns the demP estimated through the bootstrapping
    eqOPPmetrics: numpy (bins, num_rep), returns the eqOPP estimated through the bootstrapping
    eqODDmetrics: numpy (bins, num_rep), returns the eqODD estimated through the bootstrapping
    '''
    
    #Printing the f1_score and fairness metrics
    #Overall metrics:
    f1score_overall = abs(fm.f1score(y_true, y_pred))
    accuracy_overall = abs(fm.accuracy(y_true, y_pred))
    demp_overall = abs(fm.demographic_parity_dif(y_pred,prot_att,priv))
    eqOPP_overall = abs(fm.equal_opp_dif(1*(positive_class==y_true), y_pred, prot_att, priv))
    eqODD_overall = abs(fm.equalized_odd_dif(1*(positive_class==y_true), y_pred, prot_att, priv))
    
    print('f1_score: {0}'.format(f1score_overall))
    print('accuracy: {0}'.format(accuracy_overall))
    print('demP: {0}'.format(demp_overall))
    print('eqOPP: {0}'.format(eqOPP_overall))
    print('eqODD: {0}'.format(eqODD_overall))
    
    #Computing batch size for privileged and unprivileged groups
    batch_size_priv = int(sum(1*(prot_att==priv))/bins)
    batch_size_unpriv = int(sum(1*(prot_att!=priv))/bins)
    
    #Preparing data_frame by using the inputs provided
    #It used data_frame since is easier to handle the data keeping consistency
    data_frame = pd.concat([pd.DataFrame(prot_att, columns=['sensitive_att']), 
                            pd.DataFrame(y_true, columns=['label']), 
                            pd.DataFrame(risks, columns=['risk']),
                            pd.DataFrame(y_pred, columns=['pred'])], axis=1)
    
    #Create the list and array where results will be saved
    eqOPP = []
    eqODD = []
    demP = []
    
    eqOPP_ci = np.array([])
    eqODD_ci = np.array([])
    demP_ci = np.array([])
    
    cuartil = []
    
    q=1
    demmetrics = np.array([])
    eqOPPmetrics = np.array([])
    eqODDmetrics = np.array([])
    x_bins = np.array([])
    
    #Sort the values according to risk and retrieve the label, prediction and prottected attribute
    #Note these variables are used to compute fairness metrics
    label_total = 1*(positive_class==data_frame.sort_values(by=['risk'], ascending=ascending)['label'].to_numpy())
    pred_total = data_frame.sort_values(by=['risk'], ascending=ascending)['pred'].to_numpy()
    prot_att_total = data_frame.sort_values(by=['risk'], ascending=ascending)['sensitive_att'].to_numpy()
    
    #Create the lower and upper bound index computed in each bins.
    steps_priv = {}
    steps_unpriv = {}
    
    if accumulative == True:
        #The code below is used for computing the fairness metrics in the bins under
        #accumulative approach, i.e., the estimation is using individuals belonging
        #to the current bins plus the prevous
        for i in range(bins):
            #if i is the last bins, then include the rest
            if i==(bins-1):
                steps_priv[i] = (0, sum(1*(prot_att==priv)))
                steps_unpriv[i] = (0, sum(1*(prot_att!=priv)))
                break
                
            steps_priv[i] = (0, (i+1)*batch_size_priv+1)
            steps_unpriv[i] = (0, (i+1)*batch_size_unpriv+1)
    else:
        #The code below is used for computing fairness metrics in the bins under
        #individual approach, i.e., the estimation is using only the individuals
        #beloning to the bin
        for i in range(bins):
            #if i is the last bins, then include the rest
            if i==(bins-1):
                steps_priv[i] = (i*batch_size_priv, sum(1*(prot_att==priv)))
                steps_unpriv[i] = (i*batch_size_unpriv, sum(1*(prot_att!=priv)))
                break
                
            steps_priv[i] = (i*batch_size_priv, (i+1)*batch_size_priv-1)
            steps_unpriv[i] = (i*batch_size_unpriv, (i+1)*batch_size_unpriv-1)
    
    #Compute the estimation for each bin and sort them in the list and arrays created previously
    for i in range(bins):
        cuartil.append(q)
        lower_bound_priv, upper_bound_priv = steps_priv[i]
        lower_bound_unpriv, upper_bound_unpriv = steps_unpriv[i]

        label_priv = label_total[prot_att_total==priv][lower_bound_priv:upper_bound_priv]
        label_unpriv = label_total[prot_att_total!=priv][lower_bound_unpriv:upper_bound_unpriv]
        label = np.concatenate((label_priv,label_unpriv))

        pred_priv = pred_total[prot_att_total==priv][lower_bound_priv:upper_bound_priv]
        pred_unpriv = pred_total[prot_att_total!=priv][lower_bound_unpriv:upper_bound_unpriv]
        pred = np.concatenate((pred_priv, pred_unpriv))

        prot_att_priv = prot_att_total[prot_att_total==priv][lower_bound_priv:upper_bound_priv]
        prot_att_unpriv = prot_att_total[prot_att_total!=priv][lower_bound_unpriv:upper_bound_unpriv]
        protected = np.concatenate((prot_att_priv, prot_att_unpriv))

        label_priv = label_total[prot_att_total==priv][lower_bound_priv:upper_bound_priv]
        label_unpriv = label_total[prot_att_total!=priv][lower_bound_unpriv:upper_bound_unpriv]
        label = np.concatenate((label_priv, label_unpriv))           

        demP_mean, demP_std, dem_samples = estimatingFM(sub_sample_size=sample_size,
                                                        num_rep=num_rep,
                                                        nan_treatment=nan_treatment,
                                                        fm_function=fm.demographic_parity_dif, 
                                                        y_pred=pred, 
                                                        protected_attr=protected, 
                                                        priv_class=priv)
                                 
        eqOPP_mean, eqOPP_std, eqOPP_samples = estimatingFM(sub_sample_size=sample_size,
                                                            num_rep=num_rep,
                                                            nan_treatment=nan_treatment,
                                                            fm_function=fm.equal_opp_dif, 
                                                            y_true=label, 
                                                            y_pred=pred, 
                                                            protected_attr=protected, 
                                                            priv_class=priv)
                                 
        eqODD_mean, eqODD_std, eqODD_samples = estimatingFM(sub_sample_size=sample_size, 
                                                            num_rep=num_rep, 
                                                            nan_treatment=nan_treatment,
                                                            fm_function=fm.equalized_odd_dif, 
                                                            y_true=label, 
                                                            y_pred=pred, 
                                                            protected_attr=protected, 
                                                            priv_class=priv)

        demmetrics = np.append(demmetrics, dem_samples)
        eqOPPmetrics = np.append(eqOPPmetrics, eqOPP_samples)
        eqODDmetrics = np.append(eqODDmetrics, eqODD_samples)
        x_bins = np.append(x_bins, np.ones(len(dem_samples))*q)

        demP.append(demP_mean)
        eqOPP.append(eqOPP_mean)
        eqODD.append(eqODD_mean)
        
        ci = sns.utils.ci(sns.algorithms.bootstrap(eqOPP_samples))
        eqOPP_ci = np.append(eqOPP_ci, ci, axis = 0)
        
        ci = sns.utils.ci(sns.algorithms.bootstrap(eqODD_samples))
        eqODD_ci = np.append(eqODD_ci, ci, axis = 0)
        
        ci = sns.utils.ci(sns.algorithms.bootstrap(dem_samples))
        demP_ci = np.append(demP_ci, ci, axis = 0)
        
        q+=1
    
    #Plot the results
    if accumulative==True:
        ax = sns.lineplot(x_bins, demmetrics, label = 'abs(demP)')
        ax = sns.lineplot(x_bins, eqOPPmetrics, label = 'abs(eqOPP)')
        ax = sns.lineplot(x_bins, eqODDmetrics, label = 'abs(eqODD)')
        ax.set_xlabel(f'Accumulated Bin, Ascending Sort = {ascending}')
        ax.set_xticks(np.array(cuartil))
    else:
        metrics = np.concatenate((np.tile(['abs(demP)'], x_bins.shape[0]),
                            np.tile(['abs(eqOPP)'], x_bins.shape[0]),
                            np.tile(['abs(eqODD)'], x_bins.shape[0])))
        results = np.concatenate((demmetrics,
                            eqOPPmetrics,
                            eqODDmetrics))
        x_bins = np.tile(x_bins,3).astype(int)
        
        d = pd.concat([pd.DataFrame(x_bins, columns=['bins']),
                       pd.DataFrame(results, columns=['Results']), 
                       pd.DataFrame(metrics, columns=['Metrics'])],
                      axis=1)
        ax = sns.barplot(x="bins", y="Results", hue="Metrics", data=d)

        ax.set_xlabel(f'Bin, Ascending Sort = {ascending}')
    
    ax.axhline(y=demp_overall, c =sns.color_palette()[0], ls='--')
    ax.axhline(y=eqOPP_overall, c =sns.color_palette()[1], ls='--')
    ax.axhline(y=eqODD_overall, c =sns.color_palette()[2], ls='--')
    ax.set_ylabel('Metric')
    ax.set_ylim(-0.05, 1.05)
    if bart:
        ax.set_title('S-BART')
    else:
        ax.set_title('Foresee with alpha='+str(beta))
    
    plt.tight_layout()
    plt.show()
                                 
    #Transform the output into the proper dtype
    demP_ci = demP_ci.reshape(-1,2)
    eqOPP_ci = eqOPP_ci.reshape(-1,2)
    eqODD_ci = eqODD_ci.reshape(-1,2)
    
    demmetrics = demmetrics.reshape(bins, -1)
    eqOPPmetrics = eqOPPmetrics.reshape(bins, -1)
    eqODDmetrics = eqODDmetrics.reshape(bins, -1)
                                 
    return demP, eqOPP, eqODD, demP_ci, eqOPP_ci, eqODD_ci, demmetrics, eqOPPmetrics, eqODDmetrics

def preprocessProfileDataset(profile_dataset, bins, priv, sens_attr, ascending):
    #Generate the bins feature
    #Compute size of batches
    batch_size_priv = int(sum(1*(profile_dataset[sens_attr]==priv))/bins) #remember that female=1 and male=1
    batch_size_unpriv = int(sum(1*(profile_dataset[sens_attr]!=priv))/bins)

    #Assigning bins according to the order logic given to ascending varible
    profile_dataset = profile_dataset.sort_values(by=['risk'], ascending=ascending)

    bins_unpriv = np.repeat(list(range(1,bins+1)), batch_size_unpriv)
    bins_priv = np.repeat(list(range(1,bins+1)), batch_size_priv)

    #We add the rest samples
    bins_unpriv = np.append(bins_unpriv, np.repeat(bins, sum(1*(profile_dataset[sens_attr]!=priv))-bins_unpriv.shape[0]))
    bins_priv = np.append(bins_priv, np.repeat(bins, sum(1*(profile_dataset[sens_attr]==priv))-bins_priv.shape[0]))

    #Assign bins
    profile_dataset.loc[profile_dataset[sens_attr]!=priv, 'bins']=bins_unpriv
    profile_dataset.loc[profile_dataset[sens_attr]==priv, 'bins']=bins_priv
    
    return profile_dataset

def radar_ds(profile_dataset, numbers_features, features, bins):
    #Compute the differences between decile 1 and decile bins
    #for i in range(len(profile_dataset.columns)-3):
    #    profile_dataset.iloc[:,i]=(profile_dataset.iloc[:,i]-profile_dataset.iloc[:,i].mean())/(profile_dataset.iloc[:,i].std())
     
    filtered_features = features[np.where(features!='risk')[0]]
    prof_ds =  profile_dataset.iloc[:, np.where(features!='risk')[0]]
    
    differences = abs(pd.pivot_table(prof_ds, index = ['bins'], values = filtered_features).iloc[0,:]
                      -pd.pivot_table(prof_ds, index = ['bins'], values = filtered_features).iloc[bins-1,:])
    
    #range_cols = profile_dataset.loc[:, np.array(differences.index)].describe().iloc[7,:] -profile_dataset.loc[:, np.array(differences.index)].describe().iloc[3,:]
    
    #differences = differences/range_cols

    #Obtain the list of variables with higher differences
    columns_highest_dif = np.array(differences.sort_values(ascending=False)[:numbers_features].index)
    
    #Compute the pivot_table
    ds_radarchart = pd.pivot_table(profile_dataset, index = ['bins'], values = columns_highest_dif)
    #ds_radarchart = (ds_radarchart-ds_radarchart.mean())/ds_radarchart.std()
    
    return ds_radarchart

def plotProfiles(profile_dataset, numbers_features, features, bins, priv, sens_attr, ascending):
    
    profile_dataset_sorted = preprocessProfileDataset(profile_dataset, bins, priv, sens_attr, ascending)
    ds_radarchart = radar_ds(profile_dataset_sorted, numbers_features, features, bins)
    
    categories = list(ds_radarchart.columns)
    fig = go.Figure()
    
    groups = [0, 4]
    
    
    # number of variable
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(5, 4))
    
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, fontsize=12, weight="bold", color="black")


    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.yaxis.grid(which='major', color='gray', linestyle='-', linewidth=.1)
    ax.spines["polar"].set_visible(False)
    
    # ------- PART 2: Add plots
 
    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    values=ds_radarchart.loc[1].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Low Risk")
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values=ds_radarchart.loc[5].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="High Risk")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Show the graph
    plt.show()
    
    return profile_dataset_sorted, ds_radarchart, categories

def getBins(risk, A, bins):
    '''
    Returns the bins given a dataset and the risks to each sentence.
    The bins are made by, first, split X into the different groups in A.
    Then, each sub-sample is sorted by the risk in ascending order. Third,
    each sub-sample is divided into b bins of equal size. And finally, each
    corresonding bin is merged again. This way the final bins will have similar
    distribution of A.
    
    Assumptions:
    - risk and A are logically related. This means, that the first element of risk
    corresponds to the same instance of A and so on.
    - The assignment of bins are returned to each instance according to the order provided
    in risk and A.
    
    Input:
    risk: numpy (n,1), containing the risks for each sample to divide into b bins
    A: numpy (n,1), containing the sensitive attributes used to group instances
    bins: int, indicating the number of bins to divide the instances
    
    Output:
    assignment: numpy (n,1), containing the bins assigned to each instance
    '''
    
    indices = np.indices(A.shape)[0]
    merge_all = np.concatenate((indices.reshape(-1,1), risk.reshape(-1,1), A.reshape(-1,1)), axis=1)
    merge_all = merge_all[merge_all[:,1].argsort()]

    groups = {}
    unique_groups = np.unique(A)

    for g in unique_groups:
        group = merge_all[A==g]
        size_bin = int(group.shape[0]/bins)

        assignment = np.array(())

        for i in range(bins):
            if i+1==bins:
                assignment = np.concatenate((assignment, np.repeat(i+1,size_bin+group.shape[0]-size_bin*bins)))
            else:
                assignment = np.concatenate((assignment, np.repeat(i+1,size_bin)))

        groups[g] = np.concatenate((group, assignment.reshape(-1,1)),axis=1)

    final = np.array(())
    for g in unique_groups:
        final = np.concatenate((final.reshape(-1,4), groups[g]))

    return final[final[:,0].argsort()][:,-1] 
