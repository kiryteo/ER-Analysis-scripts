# import muDIC as dic
#
# path = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series002_decon_converted/std/'
# img_stack = dic.image_stack_from_folder(path, file_type='.png')
# mesher = dic.Mesher()
# #
# mesh = mesher.mesh(img_stack)
#
# settings = dic.DIC_settings(img_stack, mesh)
#
# dic_job = dic.DIC_job(settings)
# results = dic_job.run()
#
# inputs = dic.DICInput(mesh, img_stack)
#
# dic_job = dic.DICAnalysis(inputs)
# res = dic_job.run()
#
# fields = dic.Fields(res)
#
# true_strain = fields.true_strain()
#
# viz = dic.Visualizer(fields,images=img_stack)
#
# viz.show(field="True strain", component = (1,1), frame = 39)


from scipy import signal
import skimage.io as io
im1 = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series002_decon_converted/std/Series002_decon_converted_t00_ch00_std.png')
im2 = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series002_decon_converted/std/Series002_decon_converted_t01_ch00_std.png')
cor = signal.correlate2d(im1, im2)
print(cor)
