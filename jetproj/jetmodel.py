from abc import ABC
import functools
from astropy import units as u, cosmology
import numpy as np
from fourier import FINUFFT_NUNU
from pyjetshow import get_i_image as get_i_image_tangled
# FIXME: Compiled for Python 3.6.


class JetModelZoom(ABC):
    cosmo = cosmology.WMAP9

    def __init__(self, t, l, ab_fl, an_fl, nu, redshift, n_along, n_across, lg_pixel_size_mas_min, lg_pixel_size_mas_max, stokes="I",
                 tau_max=10**5, central_vfield=True, tangled_bfield=True, ft_class=FINUFFT_NUNU):
        self.t = t
        self.l = l
        self.ab_fl = ab_fl
        self.an_fl = an_fl
        self.nu = nu
        self.z = redshift
        self.stokes = stokes
        self.lg_pixel_size_mas_min = lg_pixel_size_mas_min
        self.lg_pixel_size_mas_max = lg_pixel_size_mas_max
        self.n_along = n_along
        self.n_across = n_across
        resolutions = np.logspace(lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along)
        # 2D array of u.angle pixsizes
        self.pixsize_array = np.tile(resolutions, n_across).reshape(n_across, n_along).T*u.mas

        self.stokes = stokes
        self.ft_class = ft_class
        self.ft_instance = None
        self.calculate_grid()
        self._image = None
        self.tau_max = tau_max
        self.central_vfield = central_vfield
        self.tangled_bfield = tangled_bfield
        if tangled_bfield:
            self.get_i_image = get_i_image_tangled
        else:
            raise Exception("Only random B field...")
        self._image_i = None
        self._image_tau = None
        self._updated_params = False

    def plot_resolutions(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1)
        axes.plot(np.cumsum(self.pixsize_array[:, 0]),
                  np.cumsum(self.pixsize_array[:, :int(self.n_across/2)], axis=1)[:, -1])
        axes.set_xlabel("Along distance, mas")
        axes.set_ylabel("Across distance, mas")
        axes.set_aspect("equal")
        return fig

    def halfphi_app_max(self):
        """
        Maximum half opening-angle [rad] that can be imaged with given resolution.
        """
        return np.arctan(np.sum(self.pixsize_array[-1, :int(self.n_across/2)]) / np.sum(self.pixsize_array[:, 0]))

    def calculate_grid(self):
        """
        Calculate grid of ``(r_ob, d)`` - each point is in the center of the
        corresponding pixel. Thus, ``JetModelZoom.ft`` method should not shift
        phases on a half of a pixel. ``r_ob`` and ``d`` are in parsecs.
        """
        pc_x = np.cumsum(self.pix_to_pc, axis=0)-self.pix_to_pc/2
        pc_y_up = self.pix_to_pc[:, :self.pix_to_pc.shape[1]//2][::-1]
        pc_y_low = self.pix_to_pc[:, self.pix_to_pc.shape[1]//2:]
        pc_y_up = (np.cumsum(pc_y_up, axis=1) - pc_y_up/2)[::-1]
        pc_y_low = np.cumsum(pc_y_low, axis=1) - pc_y_low/2
        pc_y = np.hstack((pc_y_up[:, ::-1], -pc_y_low))
        self.r_ob = pc_x
        # FIXME: In analogy with older convention we need ``-`` here
        self.d = -pc_y

    
    @property
    def pc_to_mas(self):
        return (u.pc/self.ang_to_dist).to(u.mas)

    @property
    def r_ob_mas(self):
        return self.r_ob*self.pc_to_mas

    @property
    def d_mas(self):
        return self.d*self.pc_to_mas

    @property
    def imgsize(self):
        return np.max(np.cumsum(self.pixsize_array, axis=0)), \
               np.max(np.cumsum(self.pixsize_array, axis=1))

    @property
    def img_extent(self):
        s0, s1 = self.imgsize[0].value, self.imgsize[1].value
        return 0, s0, -s1/2, s1/2

    @property
    def nparams(self): return 11

    def set_params_vec(self, vec):
        self.dx, self.dy, self.rot, self.theta, self.phi, B1, K1, Gamma, m, n, s, ds = vec
        self.B1 = np.exp(B1)
        self.K1 = np.exp(K1)
        self.gamma = np.exp(Gamma)
        self.m = np.exp(m)
        self.n = np.exp(n)
        self.s = np.exp(s)
        self.ds = np.exp(ds)
        self._updated_params = True

    def run(self):
        result = self.get_i_image(self.t, self.l, self.ab_fl, self.an_fl,self.theta, self.z, self.n_along, self.n_across,
                                                          10**self.lg_pixel_size_mas_min, 10**self.lg_pixel_size_mas_max,
                                                          self.phi, self.B1, self.m, self.K1, self.n, self.s,self.ds, self.gamma,
                                                          (self.nu/u.GHz).value, self.tau_max, self.central_vfield)
        self._image_tau = np.atleast_2d(result[0])
        self._image_i = np.atleast_2d(result[1])
        self._updated_params = False

    def image(self):
        if self._updated_params:
            print("Parameters are updated => running transfer")
            self.run()
            print("Finished Jetodel.run")
        else:
            print("Parameters are the same => returning old image")
        return self._image_i
    
    def image_intensity(self):
        # Factor that accounts non-uniform pixel size in plotting
        factor = (self.pixsize_array/np.min(self.pixsize_array))**2
        return self.image()/factor.T

    def image_tau(self):
        if self._updated_params:
            self.run()
        return self._image_tau

    def ft(self, uv):
        print("Beginning JetModel.ft")
        mas_to_rad = u.mas.to(u.rad)
        rot = np.array([[np.cos(self.rot), -np.sin(self.rot)],
                        [np.sin(self.rot), np.cos(self.rot)]])

        # No need in half pixel shifts cause coordinates are already at pixel
        # centers
        shift = [self.dx*mas_to_rad, self.dy*mas_to_rad]
        result = np.exp(-2.0*np.pi*1j*(uv @ shift))
        uv = uv @ rot

        x = (self.d*u.pc/self.ang_to_dist).to(u.rad).value
        y = (self.r_ob*u.pc/self.ang_to_dist).to(u.rad).value
        ft_instance = self.ft_class(uv, x.ravel(), y.ravel())
        img = self.image()
        result *= ft_instance.forward(img.T.ravel())
        print("Done JetModel.ft")
        del ft_instance, x, y

        return result

    @property
    @functools.lru_cache()
    def ang_to_dist(self):
        return self.cosmo.kpc_proper_per_arcmin(self.z)

    @property
    @functools.lru_cache()
    def pix_to_pc(self):
        """
        2D array of pixel sizes in parsecs.
        """
        return (self.pixsize_array * self.ang_to_dist).to(u.pc).value

    def plot_contours(self, nlevels=15, outfile=None):
        # Factor that accounts non-uniform pixel size in plotting
        # FIXME: self.image have fluxes already fixed in C++ code?
        factor = (self.pixsize_array/np.min(self.pixsize_array))**2
        image = self.image()/factor.T
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, axes = plt.subplots(1, 1)
        image = np.ma.array(image, mask=image == 0)
        levels = MaxNLocator(nbins=nlevels).tick_values(image.min(),
                                                        image.max())
        cmap = plt.get_cmap('plasma')

        # Contours are *point* based plots, so it is suitable for ``d`` and
        # ``r_ob`` that are centers of pixels.
        cf = axes.contourf(self.r_ob, self.d, image.T, levels=levels, cmap=cmap)
        axes.set_ylabel(r"$d$, pc")
        axes.set_xlabel(r"$r_{\rm ob}$, pc")

        # Make a colorbar with label and units
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="10%", pad=0.00)
        cb = fig.colorbar(cf, cax=cax)
        # Intensity is in Jy per minimal pixel
        cb.set_label("Intensity, Jy/pix")
        if outfile:
            fig.savefig(outfile, dpi=600, bbox_inches="tight")
        return fig

    # FIXME: This does not work anymore
    def plot(self, nlevels=15, outfile=None):
        # Factor that accounts non-uniform pixel size in plotting
        factor = (self.pixsize_array/np.min(self.pixsize_array))**2
        image = self.image()/factor.T
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from matplotlib.colors import BoundaryNorm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, axes = plt.subplots(1, 1)
        image = np.ma.array(image, mask=image == 0)
        levels = MaxNLocator(nbins=nlevels).tick_values(image.min(),
                                                        image.max())
        cmap = plt.get_cmap('plasma')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Here X and Y are 2D arrays of bounds, so ``image`` should be the value
        # *inside* those bounds. Therefore, we should remove the last value from
        # the ``image`` array. Currently we are not doing it.
        im = axes.pcolormesh(self.r_ob, self.d, image.T, norm=norm, cmap=cmap)
        axes.set_ylabel(r"$d$, pc")
        axes.set_xlabel(r"$r_{\rm ob}$, pc")

        # Make a colorbar with label and units
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="10%", pad=0.00)
        cb = fig.colorbar(im, cax=cax)
        # Intensity is in Jy per minimal pixel
        cb.set_label("Intensity, Jy/pix")
        if outfile:
            fig.savefig(outfile, dpi=600, bbox_inches="tight")
        return fig


if __name__ == "__main__":

    images = list()
    freqs = [15.4, 8.1]
    for freq in freqs:

        jm = JetModelZoom(30.0, 0.1, 0.3, 3, freq*u.GHz, 0.0165, 500, 60, -3, -1, central_vfield=False)
        jm.set_params_vec(np.array([0.0, 0.0, 0.0, 1.57, 0.068, np.log(0.1), np.log(10000), np.log(2.0), np.log(1.0),
                                    np.log(2.0), np.log(2.5), np.log(0.1)]))
        images.append(jm.image())

    alpha = np.log(images[1]/images[0])/np.log(freqs[1]/freqs[0])
    import matplotlib.pyplot as plt
    plt.matshow(alpha)
    plt.colorbar()

    import matplotlib.pyplot as plt
    fig = jm.plot_contours()
    jm.plot_resolutions()
    plt.show()