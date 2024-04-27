# dMWRIPrep
- anat-only pipeline works without installing FreeSurfer
- doing dwi preproc needs FreeSurfer at least for one step
    - the `mri_robust_template` function in FS
    - used by `niworkflows.interfaces.freesurfer.StructuralReference`
    - called by `smriprep.workflows.anatomical.init_anat_template_wf`
    - called by `smriprep.workflows.anatomical.init_anat_fit_wf`
    - called by `smriprep.workflows.anatomical.init_anat_preproc_wf`