import torch

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

device = "cpu"


def eval_modelCNN(model,test_loader,loss_module):
    #Evaluate simple CNN
    model.eval()
    with torch.no_grad():
        total_test_loss = 0.0
        for test_data_inputs, test_data_labels in test_loader:
            test_data_inputs = test_data_inputs.to(device)
            test_data_labels = test_data_labels.to(device).long()

            test_preds = model(test_data_inputs).squeeze(dim=1)
            test_loss = loss_module(test_preds, test_data_labels)

            total_test_loss += test_loss.item()

        average_test_loss = total_test_loss / len(test_loader)

    print(f'Test Loss: {average_test_loss:.4f}')

    return average_test_loss


def eval_modelAE(model,test_loader,loss_module):
    model.eval()
    threshold = 0.02
    accurate_reconstructions = 0
    total_samples = 0
    with torch.no_grad():
            total_test_loss = 0.0
            for test_data_inputs, _ in test_loader:
                test_data_inputs = test_data_inputs.to(device)
                test_preds = model(test_data_inputs)
                test_loss = loss_module(test_preds, test_data_inputs)
                #test_loss = test_loss.mean(dim=(1, 2, 3)) 
                total_test_loss += test_loss.sum().item()
                # accurate_reconstructions += (test_loss < threshold).sum().item()
                #total_samples += test_loss.size(0)
    average_test_loss = total_test_loss / len(test_loader)
    # accuracy = accurate_reconstructions / total_samples
    print(f'Test Loss: {average_test_loss:.4f}')
    # print(f'Accuracy: {accuracy:.2%}')

    return average_test_loss

def eval_modelVAE(model,test_loader,loss_module):
    model.eval()
    with torch.no_grad():
        total_test_loss = 0.0
        for test_data_inputs, test_data_labels in test_loader:
            test_data_inputs = test_data_inputs.to(device)
            test_preds, mean, log_var = model(test_data_inputs)#.squeeze(dim=1)
            test_loss, _, _  = loss_module(test_preds, test_data_inputs, mean, log_var)
            total_test_loss += test_loss
        average_test_loss = total_test_loss / len(test_loader)

    print(f'Test Loss: {average_test_loss:.4f}')


def eval_modelAEEquations(model,test_loader,loss_module):
    total_loss = 0
    processed = 0
    for data in test_loader:
        data = data.float().to(device)
        #print(data.shape)
        reconstructed = model(data)
        loss = loss_module(reconstructed, data)
        total_loss += loss.item()
        processed += len(data)
        print(f'Test Loss: {total_loss/processed}')
    
    return {total_loss/processed}

def eval_modelVAEEquations(model,test_loader,loss_module):
    total_loss = 0
    processed = 0
    for data in test_loader:
        data = data.float().to(device)
        #print(data.shape)
        reconstructed, _, _ = model(data)
        loss = loss_module(reconstructed, data)
        total_loss += loss.item()
        processed += len(data)
        print(f'Test Loss: {total_loss/processed}')
    
    return {total_loss/processed}