#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from fcnn_model import *

model_dim = "2D"
twelve_lead_model_filename = '12_lead_model.sav'
six_lead_model_filename = '6_lead_model.sav'
three_lead_model_filename = '3_lead_model.sav'
two_lead_model_filename = '2_lead_model.sav'
if model_dim == "1D":
    classifier_loader = get_model_cnc
elif model_dim == "2D":
    classifier_loader = get_model_base_2d
mat_size = {'1D':2, '2D':3}

################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    data_dict = {}
    label_dict = {}
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')

    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)
    print(num_classes, classes)

    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    # data = np.zeros((num_recordings, 14), dtype=np.float32) # 14 features: one feature for each lead, one feature for age, and one feature for sex
     # One-hot encoding of classes

    for i in range(num_recordings):
        try:
            print('    {}/{}...'.format(i+1, num_recordings))

            # Load header and recording.
            header = load_header(header_files[i])
            recording = load_recording(recording_files[i])
        
            # Get age, sex and root mean square of the leads.
            # age, sex, sig = get_features(header, recording, twelve_leads)
            # data[i, 0:12] = rms
            # data[i, 12] = age
            # data[i, 13] = sex
        
            
            
            signal = get_features(header, recording, twelve_leads)
            if model_dim == "2D":
                signal = np.expand_dims(signal,0)
            feat_shape = signal.shape
            data = signal
            if feat_shape[-1] not in data_dict.keys():
                data_dict[feat_shape[-1]] = data
            else:
                # pdb.set_trace()
                data_dict[feat_shape[-1]] = np.concatenate([data_dict[feat_shape[-1]], data], axis=0)
            
            labels = np.zeros((feat_shape[0], num_classes), dtype=np.bool)
            current_labels = get_labels(header)
            for label in current_labels:
                if label in classes:
                    j = classes.index(label)
                    labels[:, j] = 1
            if feat_shape[-1] not in label_dict.keys():
                label_dict[feat_shape[-1]] = labels
            else:
                label_dict[feat_shape[-1]] = np.concatenate([label_dict[feat_shape[-1]], labels], axis=0)       
                # len_data += len_feat
        except Exception as e :
            
            print(e)
            continue
                
    # Train models.

    # Define parameters for random forest classifier.
    n_estimators = 3     # Number of trees in the forest.
    max_leaf_nodes = 100 # Maximum number of leaf nodes in each tree.
    random_state = 0     # Random state; set for reproducibility.
    EPOCHS = 3
    N = 5000
    BATCH = 128
    # Train 12-lead ECG model.
    lead_dict = {12:twelve_leads, 6:six_leads, 3:three_leads, 2:two_leads}
    model_file_name_dict = {12:twelve_lead_model_filename, 6: six_lead_model_filename, 3:three_lead_model_filename, 2:two_lead_model_filename}
    imputer = SimpleImputer()
    for ld in [12,6,3,2]:
        
        print('Training {}-lead ECG model...'.format(ld))
        leads = lead_dict[ld]
        filename = os.path.join(model_directory, model_file_name_dict[ld])
        feature_indices = [twelve_leads.index(lead) for lead in leads]
        # features = data#[:, feature_indices]
        classifier = classifier_loader(num_classes = num_classes)
        for length in data_dict.keys():
            try:
                feature_all = data_dict[length]
                print(len(feature_all)%len(leads))
                sel_index = [i for i in range(feature_all.shape[mat_size[model_dim]-2]) if i%12 in feature_indices]
                if model_dim == "1D":
                    feature = feature_all[sel_index,:]
                elif model_dim == "2D":
                    feature = feature_all[:,sel_index,:]
    
                feature = np.expand_dims(feature,mat_size[model_dim])
                # feature = feature[:,:N,:]
                
                labels = label_dict[length]
                if model_dim == "1D":
                    labels = labels[sel_index,:]
                elif model_dim == "2D":
                    labels = labels
                
                print(feature.shape)
                # feature[feature==np.nan] = 0
                
                
                classifier.fit(feature, labels, batch_size = BATCH,epochs=EPOCHS)
            except Exception as e:
                print(e)
                continue
    
        save_model(filename, classes, leads, imputer, classifier)

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    d = {'classes': classes, 'leads': leads, 'imputer': imputer}
    joblib.dump(d, filename, protocol=0)
    model_path = filename.replace(".sav",".hdf5")
    classifier.save_weights(model_path)

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    content_dict = load_model(filename)
    num_classes = len(content_dict['classes'])
    classifier = classifier_loader(num_classes)
    classifier.load_weights(filename.replace(".sav",".hdf5"))
    content_dict['classifier']=classifier
    return content_dict

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    content_dict = load_model(filename)
    num_classes = len(content_dict['classes'])
    classifier = classifier_loader(num_classes)
    classifier.load_weights(filename.replace(".sav",".hdf5"))
    content_dict['classifier']=classifier
    return content_dict

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    content_dict = load_model(filename)
    num_classes = len(content_dict['classes'])
    classifier = classifier_loader(num_classes)
    classifier.load_weights(filename.replace(".sav",".hdf5"))
    content_dict['classifier']=classifier
    return content_dict

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    content_dict = load_model(filename)
    num_classes = len(content_dict['classes'])
    classifier = classifier_loader(num_classes)
    classifier.load_weights(filename.replace(".sav",".hdf5"))
    content_dict['classifier']=classifier
    return content_dict

# Generic function for loading a model.
def load_model(filename):
    return joblib.load(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording):
    classes = model['classes']
    leads = model['leads']
    imputer = model['imputer']
    classifier = model['classifier']
    data_dict = {}
    # Load features.
    num_leads = len(leads)
    data = np.zeros(num_leads+2, dtype=np.float32)
    # age, sex, rms = get_features(header, recording, leads)
    # data[0:num_leads] = rms
    # data[num_leads] = age
    # data[num_leads+1] = sex
    
    
    
    signal = get_features(header, recording, leads)
    if model_dim == "2D":
        signal = np.expand_dims(signal,0)
    feat_shape = signal.shape
    data = signal
    if feat_shape[-1] not in data_dict.keys():
        data_dict[feat_shape[-1]] = data
    else:
        data_dict[feat_shape[-1]] = np.concatenate([data_dict[feat_shape[-1]], data], axis=0)
    
    lead_dict = {12:twelve_leads, 6:six_leads, 3:three_leads, 2:two_leads}
    model_file_name_dict = {12:twelve_lead_model_filename, 6: six_lead_model_filename, 3:three_lead_model_filename, 2:two_lead_model_filename}
    ld = len(leads)


      
    feature_indices = [twelve_leads.index(lead) for lead in leads]
    
    
    
    label_array = None
    prob_array = None
    for length in data_dict.keys():
        try:
            feature = data_dict[length]
            # print(len(feature_all)%len(leads))
            # sel_index = [i for i in range(feature_all.shape[mat_size[model_dim]-2]) if i%12 in feature_indices]
            # if model_dim == "1D":
                # feature = feature_all[sel_index,:]
            # elif model_dim == "2D":
                # feature = feature_all[:,sel_index,:]
            # feature = imputer.transform(feature)
            feature = np.expand_dims(feature,mat_size[model_dim])
    
            print(feature.shape)
            labels = classifier.predict(feature)
            labels = (labels > 0.2 + 0)
            labels = np.asarray(labels, dtype=np.int)
            probabilities = classifier.predict(feature)
            probabilities = np.asarray(probabilities, dtype=np.float32)
            if label_array is None:
                label_array = labels
            else:
                label_array = np.concatenate([label_array,labels], axis = 0)
            if prob_array is None:
                prob_array = probabilities
            else:
                prob_array = np.concatenate([prob_array,probabilities], axis = 0)
        except Exception as e :
             print(e)
             continue

    # Impute missing data.
    # features = data.reshape(1, -1)
    # features = imputer.transform(features)

    # Predict labels and probabilities.
    # labels = classifier.predict(features)
    # labels = np.asarray(labels, dtype=np.int)[0]

    
    r_label = np.max(label_array,0)
    r_prob = np.mean(prob_array,0)
    print(r_label, '\n', r_prob)
    return classes,r_label , r_prob

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # # Compute the root mean square of each ECG lead signal.
    # sig = np.zeros(recording.shape[1], dtype=np.float32)
    # for i in range(num_leads):
    #     x = recording[i, :]
    #     sig[i] = x

    return recording

