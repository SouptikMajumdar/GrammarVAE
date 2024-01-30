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

#device = "cpu"


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
        reconstructed, mean, log_var = model(data)
        loss, recon_loss, kd = loss_module(reconstructed, data, mean, log_var)
        total_loss += loss
        processed += len(data)
        #print(f'Test Loss: {total_loss/processed}')
    
    return {total_loss/processed}


def sample_fromGVAE(model,test_loader,masks_tensor,ind_of_ind_tensor,max_length,num_rules,emb):
    ind_of_ind_tensor = ind_of_ind_tensor.to(device)
    masks_tensor = masks_tensor.to(device)
    def mask_prob(x):
        x_one_hot_decode = torch.argmax(x,dim=-1) #BatchX16
        x_one_hot_decode = x_one_hot_decode.reshape(-1)
        x_one_hot_decode = x_one_hot_decode.to(device)
        idx = torch.index_select(ind_of_ind_tensor,0,x_one_hot_decode).unsqueeze(1) #Shape 1XBatchSize*16
        idx = idx.long()
        masks = masks_tensor[idx] #Shape Batch*16X1X16
        masks = masks.view(-1,max_length,num_rules) #Shape BatchX16X16
        m_prob =  torch.mul(torch.exp(x),masks)
        m_prob = torch.div(m_prob,torch.sum(m_prob,dim=-1,keepdim=True)) #Normalize
        return m_prob
    
    def get_RHS(production):
        rhs = production.split(' -> ')[1] # Split on spaces to get individual symbols
        rhs = rhs.replace("'","").split()
        non_terminals = [symbol for symbol in rhs if symbol in ['S', 'T']]  # Filter non-terminals
        return non_terminals

    equations_decoded = []
    equations_actual = []
    recon_losses = []
    criterion = torch.nn.BCELoss(reduction='sum')
    for sample in test_loader:
        model.eval()
        sample = sample.float().to(device)
        recon, mean, log_var = model(sample)
        probs = mask_prob(recon)
        for i,ele in enumerate(recon):
            equation = []
            stack = ['S']
            t = 0
            recon_loss = criterion(recon[i],sample[i])
            while stack and t<16:
                cur = stack.pop()
                if cur in ['S', 'T']: 
                    rule_prob = probs[i, t] #.cpu().detach().numpy()   
                    #rule = torch.multinomial(rule_prob, 1, replacement=True).item()
                    rule = torch.argmax(rule_prob)
                    production = emb.idx_to_rule[rule.item()]

                    non_terminals = get_RHS(production)

                    stack.extend(non_terminals[::-1])

                    equation.append(production)
                    #print(equation)
                else:
                    pass
                t = t + 1
            
            recon_losses.append(recon_loss)
            equations_decoded.append(equation)

    for sample in test_loader:
        sample = sample.float().to(device)
        for i,ele in enumerate(sample):
            equation = []
            stack = ['S']
            t = 0
            while stack and t<16:
                cur = stack.pop()
                if cur in ['S', 'T']: 
                    rule_prob = sample[i, t] #.cpu().detach().numpy()   
                    #rule = torch.multinomial(rule_prob, 1, replacement=True).item()
                    rule = torch.argmax(rule_prob)
                    production = emb.idx_to_rule[rule.item()]
                    non_terminals = get_RHS(production)
                    stack.extend(non_terminals[::-1])
                    equation.append(production)
                    #print(equation)
                else:
                    pass
                t = t + 1
            equations_actual.append(equation)
    return equations_decoded, equations_actual, recon_losses
    
