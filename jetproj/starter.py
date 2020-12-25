import os
import sys
import numpy as np
import astropy.units as u
from .jetmodel import JetModelZoom
from data import Data
import json
import matplotlib
import matplotlib.pyplot as plt

# git clone https://ipashchenko@bitbucket.org/ipashchenko/ve.git
# FIXME: Substitute path to ve/vlbi_errors
sys.path.insert(0, '/home/rustam/github/ve/vlbi_errors')
from spydiff import clean_difmap, modelfit_core_wo_extending
from from_fits import (create_model_from_fits_file,
                       create_clean_image_from_fits_file)
from image_ops import rms_image
from image import find_bbox
from image import plot as iplot
from components import DeltaComponent, CGComponent, EGComponent
from spydiff import modelfit_difmap, import_difmap_model


def jet_modeling(flare_profile=1.3, b_ampl=0.95, n_ampl=1000, start_epoch=-10, finish_epoch=200,
                 data_dir="/home/rustam/easy_jet/data/", frequencies=[8.1, 2], B=1, N=10000.0, redshift=2.286):
    """Simulates a jet with a flare on two frequencies
    Function creates a jet model, connect with difmap,
    substitute observed uv-data with visibilities predicted by our, adds noise, connects with difmap
    After all it finds jet parameters, and makes CLEAN-procedure
    
    flare_profile - profile of jet flare in pc
    b_ampl - amplitude of magnetic field reduction
    n_ampl - amplitude of the increase in the number of radiating particles
    B - initial value of the magnetic field
    N - initial value of number of radiating particles"""
    
    # FITS file with template data (real observations)
    observed_fits_file = os.path.join(data_dir, "J2038+5119_X_2016_04_19_pet_vis.fits")
    for frequency in frequencies:
        blc = None
        trc = None
        fl = []
        r_core = []
        time = []

        out_dir = data_dir

        for j in range(start_epoch, finish_epoch, 5):
            # Create our model and set its parameters
            jm = JetModelZoom(0.0 + j, flare_profile, b_ampl, n_ampl, frequency * u.GHz, redshift, 500, 60,
                              np.log(0.001 * 15. / frequency),
                              np.log(0.1 * 15. / frequency), central_vfield=False)

            jm.set_params_vec(np.array(
                [0.0, 0.0, 0.0, 0.086, 0.068, np.log(B), np.log(N), np.log(8.0), np.log(1), np.log(2), np.log(2.),
                 np.log(0.00001)]))
            a = jm.image()
            plt.imshow(a, cmap='hot')
            plt.savefig('hotmap{id}.png'.format(id=j))

            # Substitute real (observed) uv-data with visibilities predicted by our jet model
            data_u = Data(observed_fits_file)
            data_u._stokes = "I"

            data_u.substitute_with_model(jm)

            # Add noise (the same as in the original real data)
            data_u.add_original_noise()
            out_uvfits = "artificial.uvf"
            out_ccfits = "artificial_cc.fits"
            # Save to disk in standard UVFITS file
            data_u.save(os.path.join(out_dir, out_uvfits), downscale_by_freq=True)

            # Find parameters of the core
            # beam_fractions = np.linspace(0.5, 1.5, 11)
            modelfit_difmap(out_uvfits, "in.mdl",
                            "out.mdl",
                            path=out_dir, mdl_path=out_dir,
                            out_path=out_dir, niter=100,
                            stokes="i",
                            show_difmap_output=True)

            # # Flux of the core
            # flux = np.median([results[frac]['flux'] for frac in beam_fractions])
            #
            # # Position of the core
            # r = np.median([np.hypot(results[frac]['ra'], results[frac]['dec']) for frac in beam_fractions])

            comp = import_difmap_model("data/out.mdl")[0]
            # Position of the core
            r = np.hypot(comp.p[1], comp.p[2])
            # Flux of the core
            flux = comp.p[0]
            size = comp.p[3]

            time.append(j)
            fl.append(flux)
            r_core.append(r)

            clean_difmap(fname=out_uvfits,
                         outfname=out_ccfits, stokes="i", path=out_dir,
                         outpath=out_dir, mapsize_clean=(512, 1.5 / frequency),
                         path_to_script="/home/rustam/github/ve/difmap/final_clean_nw",
                         show_difmap_output=True)

            ccimage = create_clean_image_from_fits_file(os.path.join(out_dir, out_ccfits))
            rms = rms_image(ccimage)
            beam = ccimage.beam
            if blc is None:
                blc, trc = find_bbox(ccimage.image, 2.0 * rms, 5)
            fig = iplot(ccimage.image, x=ccimage.x, y=ccimage.y, min_abs_level=2.0 * rms,
                        beam=beam, show_beam=True, beam_place="ul", blc=blc, trc=trc,
                        close=False, colorbar_label="Jy/beam", show=False, contour_color='black',
                        fig=None, plot_colorbar=False)

            fig.savefig(os.path.join(out_dir, "model_CELAN_image" + str(int(frequency)) + "{id}.png".format(id=j)),
                        dpi=300,
                        bbox_inches="tight")

            globals()['flux' + str(int(frequency))] = np.array(fl)
            globals()['r_core' + str(int(frequency))] = np.array(r_core)
            globals()['time' + str(int(frequency))] = np.array(time)

        fig1, axes1 = plt.subplots(1, 1)
        axes1.plot(time, fl)
        axes1.set_xlabel("Time")
        axes1.set_ylabel("Flux")
        axes1.set_title("For " + str(int(frequency)) + " GHz")
        fig1.savefig('flux ' + str(int(frequency)) + ' GHz' + '.png')

        fig2, axes2 = plt.subplots(1, 1)
        axes2.plot(time, r_core)
        axes2.set_xlabel("Time")
        axes2.set_ylabel("Coordinate of core")
        axes2.set_title("For " + str(int(frequency)) + " GHz")
        fig2.savefig('position of core ' + str(int(frequency)) + ' GHz' + '.png')

        with open('time' + str(int(frequency)) + '.txt', 'w') as fw:
            json.dump(time, fw)
        with open('flux for' + str(int(frequency)) + '.txt', 'w') as fw:
            json.dump(fl, fw)
        with open('position of core for' + str(int(frequency)) + '.txt', 'w') as fw:
            json.dump(r_core, fw)

    return flux8, flux2, r_core8, r_core2, time8, time2


if __name__ == '__main__':
    jet_modeling()
