from tqdm import tqdm
import torch
import mlflow


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = "cpu"
print("Device", device)

# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#device = "cpu"


#Define Training for CNN
def train_model_CNN(model, optimizer, data_loader,val_loader, loss_module, num_epochs=100):
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        total_val_loss = 0.0
        for data_inputs, data_labels in data_loader:

            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            #data_labels = data_labels.long()
            data_labels = data_labels.to(device).long()
            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels)
            total_val_loss += loss.item()
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()
            ## Step 5: Update the parameters
            optimizer.step()
        average_train_loss = total_val_loss/len(train_loader)

    # Validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for val_data_inputs, val_data_labels in val_loader:
                val_data_inputs = val_data_inputs.to(device)
                val_data_labels = val_data_labels.to(device).long()

                val_preds = model(val_data_inputs).squeeze(dim=1)
                val_loss = loss_module(val_preds, val_data_labels)

                total_val_loss += val_loss.item()

            average_val_loss = total_val_loss / len(val_loader)

        model.train()

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')

    return model, average_train_loss, average_val_loss


#Define Training for AE
def train_AEmodel(AEmodel,train_loader,val_loader, optimizer, loss_module, num_epochs=10):
    # Set model to train mode
    AEmodel.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        total_train_loss = 0.0
        for data_inputs, data_labels in train_loader:
            data_inputs = data_inputs.to(device)
            preds = AEmodel(data_inputs)
            loss = loss_module(preds, data_inputs)
            total_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_train_loss = total_train_loss/len(train_loader)

    # Validation
        AEmodel.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for val_data_inputs, val_data_labels in val_loader:
                val_data_inputs = val_data_inputs.to(device)
                val_preds = AEmodel(val_data_inputs)#.squeeze(dim=1)
                val_loss = loss_module(val_preds, val_data_inputs)

                total_val_loss += val_loss.item()

            average_val_loss = total_val_loss / len(val_loader)

        AEmodel.train()

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')

    
    return average_train_loss, average_val_loss


def train_VAEmodel(model,train_loader,val_loader, optimizer,loss_module, num_epochs=100):
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        total_train_loss = 0.0
        for data_inputs, data_labels in train_loader:
            data_inputs = data_inputs.to(device)
            preds, mean, log_var = model(data_inputs)
            #print(preds.shape)
            loss, rc_loss, kl_loss = loss_module(preds, data_inputs,mean, log_var)
            #print(loss, rc_loss, kl_loss)
            total_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(loss, rc_loss, kl_loss)
        average_train_loss = total_train_loss/len(train_loader)

    # Validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for val_data_inputs, val_data_labels in val_loader:
                val_data_inputs = val_data_inputs.to(device)
                val_preds, mean, log_var = model(val_data_inputs)#.squeeze(dim=1)
                val_loss, _, _ = loss_module(val_preds, val_data_inputs, mean, log_var)
                total_val_loss += val_loss
            average_val_loss = total_val_loss / len(val_loader)

        model.train()

        print(f'RC Loss: {rc_loss}')
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')

    return average_train_loss, average_val_loss


def train_EqnAE(model,one_hot_encoded_training_loader,one_hot_encoded_valid_loader,loss_module,optimizer,num_epochs=100):
    # Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_patience = 5
    for epoch in range(num_epochs):
        train_loss = 0
        processed = 0
        val_loss = 0
        model.train()
        for batch_idx, data in enumerate(one_hot_encoded_training_loader):
            data = data.float().to(device)
            optimizer.zero_grad()
            recon = model(data)
            loss= loss_module(recon, data)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            processed += len(data)
        
        avg_train_loss = train_loss / processed
        print(f'====> Epoch: {epoch} Average Training loss: {train_loss / processed:.8f}')

        processed = 0
        model.eval()
        for batch_idx, data in enumerate(one_hot_encoded_valid_loader):
            data = data.float().to(device)
            recon = model(data)
            loss= loss_module(recon, data)
            val_loss += loss.item()
            processed += len(data)
        
        avg_val_loss = val_loss / processed
        print(f'====> Epoch: {epoch} Average Validation loss: {val_loss / processed:.8f}')

        mlflow.log_metric("Train Loss", avg_train_loss)
        mlflow.log_metric("Validation Loss", avg_val_loss)

        
        if best_val_loss - avg_val_loss > 1e-05:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == n_patience:
            break
        
    return avg_train_loss,avg_val_loss  



def train_EqnVAE(model,one_hot_encoded_training_loader,one_hot_encoded_valid_loader,loss_module,optimizer,num_epochs=100):
    # Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_patience = 5
    for epoch in range(num_epochs):
        train_loss = 0
        processed = 0
        val_loss = 0
        model.train()
        for batch_idx, data in enumerate(one_hot_encoded_training_loader):
            data = data.float().to(device)
            optimizer.zero_grad()
            recon,mean,log_var = model(data)
            loss, rc_loss, kl_loss = loss_module(recon, data,mean, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            processed += len(data)
        
        avg_train_loss = train_loss / processed
        print(f'====> Epoch: {epoch} Average Training loss: {train_loss / processed:.8f}')

        processed = 0
        model.eval()
        for batch_idx, data in enumerate(one_hot_encoded_valid_loader):
            data = data.float().to(device)
            recon,mean,log_var = model(data)
            loss, rc_loss, kl_loss = loss_module(recon, data,mean, log_var)
            val_loss += loss.item()
            processed += len(data)
        
        avg_val_loss = val_loss / processed
        print(f'====> Epoch: {epoch} Average Validation loss: {val_loss / processed:.8f}')

        mlflow.log_metric("Train Loss", avg_train_loss)
        mlflow.log_metric("Validation Loss", avg_val_loss)

        
        if best_val_loss - avg_val_loss > 1e-05:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == n_patience:
            break
        
    return avg_train_loss,avg_val_loss   



def train_EqnGVAE(model,one_hot_encoded_training_loader,one_hot_encoded_valid_loader,loss_module,optimizer,num_epochs=100):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_patience = 5
    for epoch in range(num_epochs):
        train_loss = 0
        processed = 0
        val_loss = 0
        model.train()
        for batch_idx, data in enumerate(one_hot_encoded_training_loader):
            data = data.float().to(device)
            optimizer.zero_grad()
            recon,mean,log_var = model(data)
            loss, rc_loss, kl_loss = loss_module(recon, data,mean, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            processed += len(data)
        
        avg_train_loss = train_loss / processed
        print(f'====> Epoch: {epoch} Average Training loss: {train_loss / processed:.8f}')

        processed = 0
        model.eval()
        for batch_idx, data in enumerate(one_hot_encoded_valid_loader):
            data = data.float().to(device)
            recon,mean,log_var = model(data)
            loss, rc_loss, kl_loss = loss_module(recon, data,mean, log_var)
            val_loss += loss.item()
            processed += len(data)
        
        avg_val_loss = val_loss / processed
        print(f'====> Epoch: {epoch} Average Validation loss: {val_loss / processed:.8f}')

        mlflow.log_metric("Train Loss", avg_train_loss)
        mlflow.log_metric("Validation Loss", avg_val_loss)

        

        
        if best_val_loss - avg_val_loss > 1e-05:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == n_patience:
            break
        
    return avg_train_loss,avg_val_loss