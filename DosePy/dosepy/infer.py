# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""Reusable inference runner.

``predict_using_model`` builds and loads the network for every patient.  When
predicting a cohort, use :class:`InferenceRunner` instead: the weights are
loaded once and only the preprocessed ``.npz`` changes.

Example::

    runner = InferenceRunner({
        "checkpoints_dir": "workspace/checkpoints",
        "results_dir": "workspace/results",
        "name": "dosepy_lung_unet128",
        "netG": "unet_128",
        "epoch": "best_mae",
        "gpu_ids": [0],
    })
    for infer_dir in prepared_infer_dirs:
        # each folder contains one <Patient>.npz produced by the preprocessing
        # path used in dosepy.preprocess.predict
        runner.run(infer_dir)
"""

import os

from dosepy.data import create_dataset
from dosepy.models import create_model
from dosepy.options import TestOptions, build_opt
from dosepy.utils import html
from dosepy.utils.visualizer import save_images

__all__ = ["InferenceRunner", "build_infer_opt"]

NPY_OUT_FNAMES = ["CT2DOSE"]


def build_infer_opt(args: dict):
    """Build inference options from a dict.

    The dataset mode is forced to ``single`` (no ground truth) and the loader
    to a deterministic batch size of one.
    """
    overrides = dict(args)
    overrides.setdefault("model", "test")
    overrides.setdefault("dataset_mode", "single")
    opt = build_opt(TestOptions, overrides)
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    return opt


class InferenceRunner:
    """Load a trained generator once and run it on many preprocessed folders."""

    def __init__(self, args: dict):
        self.opt = build_infer_opt(args)
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        if self.opt.eval:
            self.model.eval()

    def run(self, dataroot: str):
        """Run inference on every ``.npz`` under ``dataroot``.

        Returns:
            List of written prediction paths
            (``.../npz_images/<Patient>_CT2DOSE.nrrd``).
        """
        self.opt.dataroot = dataroot
        dataset = create_dataset(self.opt)

        web_dir = os.path.join(
            self.opt.results_dir,
            self.opt.name,
            "{}_{}".format(self.opt.phase, self.opt.epoch),
        )
        if self.opt.load_iter > 0:
            web_dir = "{:s}_iter{:d}".format(web_dir, self.opt.load_iter)

        print("creating web directory", web_dir)
        os.makedirs(os.path.join(web_dir, "npz_images"), exist_ok=True)

        webpage = html.HTML(
            web_dir,
            "Experiment = %s, Phase = %s, Epoch = %s"
            % (self.opt.name, self.opt.phase, self.opt.epoch),
        )

        written = []
        for i, data in enumerate(dataset):
            self.model.set_input(data)
            self.model.test()

            visuals = self.model.get_current_visuals()
            img_path = self.model.get_image_paths()
            if i % 5 == 0:
                print("processing (%04d)-th image... %s" % (i, img_path))

            save_images(
                webpage,
                visuals,
                img_path,
                aspect_ratio=self.opt.aspect_ratio,
                width=self.opt.display_winsize,
                npy_out_fnames=NPY_OUT_FNAMES,
            )

            patient = os.path.splitext(os.path.basename(img_path[0]))[0]
            written.append(os.path.join(web_dir, "npz_images", patient + "_CT2DOSE.nrrd"))

        webpage.save()
        return written
