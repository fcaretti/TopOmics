import muon as mu
import torch
from anndata import AnnData

MuDataType = mu.MuData


class BaseTopicModel:
    """
    Base class for all models in the omics_topic package.

    This class provides a common interface and shared functionality for all models.
    It can be extended by specific model implementations.
    """

    def __init__(
        self,
        mdata: MuDataType | dict[str, AnnData] | list[AnnData] | AnnData,
        modalities: list[str] | str | None = None,
    ):
        """
        Initialize the BaseModel. Checks the input data.

        Args:
            mdata: Multi-modal data container:
                - MuData object (mu.MuData),
                - dict mapping modality names to AnnData,
                - list of AnnData objects (requires `modalities`)
                - single AnnData object (requires `modalities`).
            modalities: Names corresponding to each AnnData in a list input.
        Initializes:
            - `self.data_dict`: Dictionary mapping modality names to tensors.
            - `self.modalities`: List of modality names.
            - `self.n_cells`: Number of cells (assumed to be the same across modalities).
        """
        self.check_input(mdata, modalities)

        self.check_modalities_names()

        n_cells_set = {v.shape[0] for v in self.data_dict.values()}
        if len(n_cells_set) != 1:
            raise ValueError("All modalities must share the same cells / order")

        self.n_cells = self.data_dict[self.modalities[0]].shape[0]

        print("Initializing model with the following modalities:", self.modalities)

    def fit(self, data):
        """
        Fit the model to the provided data.

        Parameters
        ----------
            data: The input data to fit the model.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def predict(self, data):
        """
        Predict using the fitted model on the provided data.

        Parameters
        ----------
            data: The input data for prediction.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def check_input(self, mdata, modalities):
        """
        Validate and process the input data.

        Checks that data are adata or mudata objects, and that the modalities are correctly specified.
        """
        if isinstance(mdata, dict):
            self.data_dict = mdata
        elif MuDataType and isinstance(mdata, MuDataType):
            self.data_dict = {mod: mdata[mod].X for mod in mdata.mod}
        elif isinstance(mdata, list):
            if modalities is None or len(modalities) != len(mdata):
                raise ValueError("When passing a list of AnnData, `modalities` must be a list of the same length.")
            if not all(isinstance(data, AnnData) for data in mdata):
                raise ValueError("All elements in the list must be AnnData objects.")
            self.data_dict = dict(zip(modalities, mdata.X, strict=False))
        elif isinstance(mdata, AnnData):
            if modalities is not None and len(modalities) != 1:
                raise ValueError("You passed a single AnnData but provided multiple modality names. ")
            if modalities is None:
                raise Warning("No modality names provided for a single AnnData. Defaulting to 'rna'.")
                self.modalities = ["rna"]
            self.data_dict = {modalities[0]: mdata.X}
            self.modalities = modalities
        else:
            raise TypeError(
                "`mdata` must be a MuData object, a dict of AnnData, a list of AnnData or a single AnnData."
            )

    def check_modalities_names(self):
        """
        Standardize and validate modality keys in data_dict.

        Maps various synonyms to 'rna', 'protein', or 'chromatin',
        and rebuilds data_dict with standardized keys.
        """
        if len(self.data_dict) == 0:
            raise ValueError("data_dict is empty. Please provide valid data.")

        # Ensure all keys are strings
        for k in self.data_dict.keys():
            if not isinstance(k, str):
                raise ValueError(f"Invalid modality key {k!r}. Must be a string.")

        # Define valid groups
        rna_syn = {"rna", "RNA", "genes", "transcripts"}
        prot_syn = {"adt", "protein", "prot", "proteins", "proteomics"}
        chrom_syn = {"chromatin", "atac"}

        seen = set()
        remap: dict[str, str] = {}
        for orig in list(self.data_dict.keys()):
            lname = orig.lower()
            if lname in rna_syn:
                std = "rna"
            elif lname in prot_syn:
                std = "protein"
            elif lname in chrom_syn:
                std = "chromatin"
            else:
                raise ValueError(
                    f"Invalid modality name '{orig}'. Must be one of rna, protein, or chromatin (or synonyms)."
                )
            if std in seen:
                raise ValueError(f"Duplicate modality '{std}' detected from key '{orig}'.")
            seen.add(std)
            remap[orig] = std

        # Rebuild dict with standardized keys
        new_dict: dict[str, torch.Tensor] = {}
        for orig, std in remap.items():
            new_dict[std] = self.data_dict[orig]
        self.data_dict = new_dict
        self.modalities = list(self.data_dict.keys())
