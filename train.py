def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # 1 step:     Clean the old gradients
        optimizer.zero_grad()

        # 2 step: Forward Pass    - calculates the predictions
        outputs= model(inputs)

        # 3 step: Calculates the error
        loss= criterion(outputs, targets)

        # 4 step: Backward Pass
        loss.backward()

        # 5 step: Update the weights
        optimizer.step()

        # Counter of metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')