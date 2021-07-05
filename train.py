#Train function
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Config
    batch_size = 4
    image_size = 256
    learning_rate = 1e-3
    beta1, beta2 = (.5, .99)
    weight_decay = 1e-3
    epochs = 100

    # Models
    netD = Discriminator().to(device)
    netG = Generator().to(device)

    optimizerD = AdamW(netD.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    optimizerG = AdamW(netG.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    # Labels
    cartoon_labels = torch.ones (batch_size, 1, image_size // 4, image_size // 4).to(device)
    fake_labels    = torch.zeros(batch_size, 1, image_size // 4, image_size // 4).to(device)

    # Loss functions
    content_loss = ContentLoss().to(device)
    adv_loss     = AdversialLoss(cartoon_labels, fake_labels).to(device)
    BCE_loss     = nn.BCELoss().to(device)

    # Dataloaders
    real_dataloader    = get_dataloader("/content/drive/MyDrive/Colab/pic2anime/trainA",           size = image_size, bs = batch_size)
    cartoon_dataloader = get_dataloader("/content/drive/MyDrive/Colab/pic2anime/trainB",        size = image_size, bs = batch_size)
    edge_dataloader    = get_dataloader("/content/drive/MyDrive/Colab/pic2anime/trainB_smooth", size = image_size, bs = batch_size)
    last_epoch = 0
    last_i = 0
    # --------------------------------------------------------------------------------------------- #
    # Training Loop
    tracked_images = next(iter(real_dataloader))[0].to(device)
    original_images = tracked_images.detach().cpu()
    grid = vutils.make_grid(original_images, padding=2, normalize=True, nrow=3)
    plt.imsave(f"/content/cartoon-gan/results/original.png", np.transpose(grid, (1,2,0)).numpy())
    netG.load_state_dict(torch.load("_trained_netG65.pth"))
    netD.load_state_dict(torch.load("_trained_netD65.pth"))
    with open("iter_data65.pickle", "rb") as handle:
      last_epoch, last_i = pickle.load(handle)

    
    # Lists to keep track of progress"
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    start_epoch = last_epoch
    start_i = last_i


    print("Starting Training Loop...")
    # For each epoch.
    for epoch in range(start_epoch,epochs):
        real_dl_iter = iter(real_dataloader)
        cartoon_dl_iter = iter(cartoon_dataloader)
        edge_dl_iter = iter(edge_dataloader)
        iterations =  min(len(real_dl_iter), len(cartoon_dl_iter))
        for i in range(start_i,iterations):
            real_data = next(real_dl_iter)
            cartoon_data = next(cartoon_dl_iter)
            edge_data = next(edge_dl_iter)
            netD.zero_grad()

            # Format batch.
            cartoon_data   = cartoon_data.to(device)
            edge_data      = edge_data.to(device)
            real_data      = real_data.to(device)

            # Generate image
            generated_data = netG(real_data)

            # Forward pass all batches through D.
            cartoon_pred   = netD(cartoon_data)      #.view(-1)
            edge_pred      = netD(edge_data)         #.view(-1)
            generated_pred = netD(generated_data)    #.view(-1)

            print(generated_data.is_cuda, real_data.is_cuda)

            errD = adv_loss(cartoon_pred, generated_pred, edge_pred)
            
            errD.backward(retain_graph=True)
            D_x = cartoon_pred.mean().item() # Should be close to 1

            optimizerD.step()
            netG.zero_grad()
            
            generated_pred = netD(generated_data) #.view(-1)

            print(generated_data.is_cuda, real_data.is_cuda)
            print("generated_pred:", generated_pred.is_cuda, "cartoon_labels:", cartoon_labels.is_cuda)
            errG = BCE_loss(generated_pred, cartoon_labels) + content_loss(generated_data, real_data)

            errG.backward()

            D_G_z2 = generated_pred.mean().item() # Should be close to 1
            
            optimizerG.step()
            
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): null / %.4f'
                    % (epoch, epochs, i, len(real_dataloader),
                        errD.item(), errG.item(), D_x, D_G_z2))
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    
                    fake = netG(tracked_images.unsqueeze(0)).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
            last_i = i
        start_i = 0
        last_epoch = epoch
        torch.save(netG.state_dict(), "_trained_netG"+str(epoch)+".pth")
        torch.save(netD.state_dict(), "_trained_netD"+str(epoch)+".pth")
        files.download("_trained_netG"+str(epoch)+".pth")
        files.download("_trained_netD"+str(epoch)+".pth")
        with open("iter_data"+str(epoch)+".pickle", "wb") as handle:
           pickle.dump([last_epoch, last_i], handle)
           print("file saved")
           files.download("iter_data"+str(epoch)+".pickle")
         
