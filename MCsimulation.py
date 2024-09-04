import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import exposure

# Simulation parameters
PartMult = 1600   # Number of particles generated in each event
kNeutralPiFrac = 0.25  # Fraction of neutral pions (particles with no charge)
kNeutralHadFrac = 0.5  # Fraction of neutral hadrons (neutral strong-interaction particles)

kMaxEta = 2.5  # Maximum pseudorapidity (used to describe the angular distribution of particles in high-energy physics)
kBField_values = [0.0, 0.5]  # List of magnetic field strengths to simulate

kNEvent = 10000   # Number of events to simulate
kImSize = 56  # Size of the output image (56x56 pixels)

kResScale = 1.0  # Resolution scaling factor
kTRKres = 0.05 * kResScale  # Tracker detector resolution
kEMCres = 0.05 * kResScale  # Electromagnetic calorimeter resolution
kHCALres = 0.1 * kResScale  # Hadronic calorimeter resolution

kNonLin = 0.0  # Non-linearity correction factor set to zero as per your requirement

def enhance_contrast(image):
    # Enhance the contrast of the image using rescale_intensity
    return exposure.rescale_intensity(image, in_range='image', out_range=(0, 255))

for kBField in kBField_values:
    fDir = f"data/Gauss_S{kResScale:.2f}_NL{kNonLin:.2f}_B{kBField:.2f}/"
    os.makedirs(fDir, exist_ok=True)
    print(f"Writing to {fDir}")

    for n in range(kNEvent):
        if n % 100 == 0:
            print(f"Generating event {n}")

        # Initialize arrays to store particle properties
        Charge = np.empty(PartMult)
        Hadron = np.empty(PartMult)
        Energy = np.empty(PartMult)
        eta = np.empty(PartMult)
        phi = np.empty(PartMult)
        WTruth = np.empty(PartMult)

        for i in range(PartMult):
            Energy[i] = abs(np.random.normal() + np.random.normal(0, 2))  # Generate random energy for each particle
            WTruth[i] = Energy[i]  # Set the true weight to be the energy of the particle

            # Randomly determine if the particle is a neutral pion, neutral hadron, or charged hadron
            x = np.random.rand()
            if x < kNeutralPiFrac:
                Charge[i] = 0
                Hadron[i] = 0
            elif x < kNeutralHadFrac:
                Charge[i] = 0
                Hadron[i] = 1
            else:
                Charge[i] = 1 if np.random.rand() > 0.5 else -1  # Randomly assign +1 or -1 charge
                Hadron[i] = 1

            # Assign pseudorapidity and azimuthal angle based on charge
            if Charge[i] == 0:
                eta[i] = (np.random.rand() - 0.5) * 2 * kMaxEta  # Larger pseudorapidity range for neutral particles
                phi[i] = (np.random.rand() - 0.5) * 2 * np.pi
            else:
                eta[i] = (np.random.rand() - 0.5) * 1.8 * kMaxEta  # Smaller range for charged particles
                phi[i] = (np.random.rand() - 0.5) * 1.8 * np.pi

        c_truth, xe, ye = np.histogram2d(eta, phi, weights=WTruth,
                                         range=[[-kMaxEta, kMaxEta], [-np.pi, np.pi]],
                                         bins=(kImSize, kImSize))

        # Tracker simulation
        kMinTrackE = 0.1
        WTrkP = np.zeros(PartMult)
        WTrkN = np.zeros(PartMult)
        for i in range(PartMult):
            if Charge[i] > 0 and Energy[i] > kMinTrackE:
                WTrkP[i] = Energy[i] * np.random.normal(1, kTRKres)
            elif Charge[i] < 0 and Energy[i] > kMinTrackE:
                WTrkN[i] = Energy[i] * np.random.normal(1, kTRKres)

        c_trkp, xe, ye = np.histogram2d(eta, phi, weights=WTrkP,
                                        range=[[-kMaxEta, kMaxEta], [-np.pi, np.pi]],
                                        bins=(kImSize, kImSize))

        c_trkn, xe, ye = np.histogram2d(eta, phi, weights=WTrkN,
                                        range=[[-kMaxEta, kMaxEta], [-np.pi, np.pi]],
                                        bins=(kImSize, kImSize))

        # Apply magnetic field effect
        for i in range(PartMult):
            if Charge[i] > 0 and Energy[i] != 0:
                phi[i] += kBField * 1 / Energy[i]  # Adjust azimuthal angle for positive particles
            elif Charge[i] < 0 and Energy[i] != 0:
                phi[i] -= kBField * 1 / Energy[i]  # Adjust for negative particles

        # Emcal simulation
        WEmcal = np.zeros(PartMult)
        kMinEmcalE = 0.2
        for i in range(PartMult):
            if Hadron[i] == 0 and Energy[i] > kMinEmcalE:
                WEmcal[i] = (Energy[i] - kNonLin * np.sqrt(Energy[i])) * 0.9 * np.random.normal(1, kEMCres)
                Energy[i] *= 0.1  # Deplete energy after calorimeter interaction
            elif Energy[i] > kMinEmcalE:
                WEmcal[i] = Energy[i] * 0.1 * np.random.normal(1, kEMCres)
                Energy[i] *= 0.9

        # Hcal simulation
        WHcal = np.zeros(PartMult)
        kMinHcalE = 0.3
        for i in range(PartMult):
            if Energy[i] > kMinHcalE:
                WHcal[i] = (Energy[i] - kNonLin * np.sqrt(Energy[i])) * 0.9 * np.random.normal(1, kHCALres)
                Energy[i] *= 0.1

        c_emcal, xe, ye = np.histogram2d(eta, phi, weights=WEmcal,
                                         range=[[-kMaxEta, kMaxEta], [-np.pi, np.pi]],
                                         bins=(kImSize, kImSize))

        c_hcal, xe, ye = np.histogram2d(eta, phi, weights=WHcal,
                                        range=[[-kMaxEta, kMaxEta], [-np.pi, np.pi]],
                                        bins=(kImSize, kImSize))

        # Enhance contrast and save images
        c_truth = enhance_contrast(c_truth)
        c_trkp = enhance_contrast(c_trkp)
        c_trkn = enhance_contrast(c_trkn)
        c_emcal = enhance_contrast(c_emcal)
        c_hcal = enhance_contrast(c_hcal)

        io.imsave(os.path.join(fDir, f"truth_{n}.tiff"), c_truth.astype(np.uint8))
        io.imsave(os.path.join(fDir, f"trkp_{n}.tiff"), c_trkp.astype(np.uint8))
        io.imsave(os.path.join(fDir, f"trkn_{n}.tiff"), c_trkn.astype(np.uint8))
        io.imsave(os.path.join(fDir, f"emcal_{n}.tiff"), c_emcal.astype(np.uint8))
        io.imsave(os.path.join(fDir, f"hcal_{n}.tiff"), c_hcal.astype(np.uint8))

        if n == 0:  # Display the first event's images
            plt.imshow(c_truth, cmap='gray')
            plt.title(f"truth (B = {kBField})")
            plt.show()
            plt.imshow(c_trkp, cmap='gray')
            plt.title(f"trkp (B = {kBField})")
            plt.show()
            plt.imshow(c_trkn, cmap='gray')
            plt.title(f"trkn (B = {kBField})")
            plt.show()
            plt.imshow(c_emcal, cmap='gray')
            plt.title(f"emcal (B = {kBField})")
            plt.show()
            plt.imshow(c_hcal, cmap='gray')
            plt.title(f"hcal (B = {kBField})")
            plt.show()
