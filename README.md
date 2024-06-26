# Getting dMRIPrep working

### TLDR
- my fork on dMRIPrep: https://github.com/man-shu/dmriprep
- current working docker image: `docker pull haggarwa/dmriprep:dwi_wo_fsrecon`
- does anatomical, diffusion preproc without freesurfer recon

### Mods
- had to remove all python package restrictions + removed default versioning system
- now runs on python 3.12 and latest deps
- external dep versions are still the same

### Notes
- anat-only pipeline works without installing FreeSurfer
- doing dwi preproc needs FreeSurfer at least for one step
    - the `mri_robust_template` function in FS
    - used by `niworkflows.interfaces.freesurfer.StructuralReference`
    - called by `smriprep.workflows.anatomical.init_anat_template_wf`
    - called by `smriprep.workflows.anatomical.init_anat_fit_wf`
    - called by `smriprep.workflows.anatomical.init_anat_preproc_wf`
- need to do `export DOCKER_DEFAULT_PLATFORM=linux/amd64` on macOS before build
- eddy step under dwi preproc needs `PhaseEncodingDirection`
    - provided via .json sidecar eg `sub-7014_dwi.json`
    - **not available for standford data**
