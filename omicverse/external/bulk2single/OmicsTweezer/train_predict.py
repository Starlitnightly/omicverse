

def train_predict(train_reference, test_bulk, num, scale, ot_weight,celltype_key='CellType'):

    from .simulation import generate_simulated_data
    from sklearn.preprocessing import MinMaxScaler
    from .deconvolution import mian
    
    ## Sampling Training Samples
    train_bulk = generate_simulated_data(train_reference,samplenum=num,sparse=False,celltype_key=celltype_key)
    if scale:
        ## Min-Max Scaling
        scaler = MinMaxScaler()
        train_bulk.X = scaler.fit_transform(train_bulk.X)
        test_bulk.X = scaler.fit_transform(test_bulk.X)


    test_bulk_copy = test_bulk.copy()
    test_bulk_copy.obs = test_bulk_copy.obs.iloc[0:test_bulk_copy.shape[0],:]
    
    ## train and predict
    predict_output, ground_truth = mian(train_bulk, test_bulk_copy,  ot_weight,sep='\t',
                               batch_size=128, epochs=30)

    return (predict_output[0]+predict_output[1]+predict_output[2])/3
