"""NITorch Registration — 3D Slicer scripted module.

Wraps the nitorch Python registration API for affine + nonlinear (SVF)
registration directly inside 3D Slicer.  Includes a validation tab for
Dice score comparison.
"""

import os
import sys
import time
import logging

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin
import qt

# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

class NITorchRegister(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "NITorch Register"
        self.parent.categories = ["Registration"]
        self.parent.dependencies = []
        self.parent.contributors = ["NITorch developers"]
        self.parent.helpText = (
            "GPU-accelerated affine + nonlinear 3D image registration powered by NITorch "
            "(https://github.com/balbasty/nitorch)."
        )
        self.parent.acknowledgementText = (
            "This module uses the NITorch library for medical image registration."
        )


# ---------------------------------------------------------------------------
# Stdout capture helper
# ---------------------------------------------------------------------------

class _LogCapture:
    """Captures stdout writes and appends them to a QPlainTextEdit.

    Filters separator lines (all dashes/equals) for cleaner display and
    batches UI updates to avoid flashing.
    """

    def __init__(self, text_widget, original_stdout):
        self._widget = text_widget
        self._original = original_stdout
        self._buffer = ""

    @staticmethod
    def _is_separator(line):
        stripped = line.strip()
        return bool(stripped) and all(c in '-=*' for c in stripped)

    def _show_line(self, line, overwrite=False):
        """Show a line in the widget. If overwrite, replace the last line."""
        if self._is_separator(line) or not line.strip():
            return
        if overwrite:
            cursor = self._widget.textCursor()
            cursor.movePosition(cursor.End)
            cursor.movePosition(cursor.StartOfBlock, cursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(line)
            self._widget.setTextCursor(cursor)
        else:
            self._widget.appendPlainText(line)

    def write(self, text):
        if not text:
            return
        # Always pass through to original stdout unfiltered
        if self._original:
            self._original.write(text)

        self._buffer += text

        # Only flush to widget on newline or carriage return
        if '\n' not in self._buffer and '\r' not in self._buffer:
            return

        self._widget.setUpdatesEnabled(False)
        try:
            # Split on \n first
            lines = self._buffer.split('\n')
            # Last element is the incomplete next line — keep in buffer
            self._buffer = lines.pop()

            for line in lines:
                # Handle \r within a line: show last segment, overwriting
                if '\r' in line:
                    parts = line.split('\r')
                    for part in parts:
                        if part.strip():
                            self._show_line(part, overwrite=True)
                else:
                    self._show_line(line)

            # If buffer contains \r (no \n yet), show latest segment
            if '\r' in self._buffer:
                parts = self._buffer.split('\r')
                self._buffer = parts.pop()  # keep only the latest incomplete part
                for part in parts:
                    if part.strip():
                        self._show_line(part, overwrite=True)

        finally:
            self._widget.setUpdatesEnabled(True)

        scrollbar = self._widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum)
        slicer.app.processEvents()

    def flush(self):
        # Flush any remaining buffer content
        if self._buffer.strip() and not self._is_separator(self._buffer):
            self._widget.appendPlainText(self._buffer)
            scrollbar = self._widget.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum)
        self._buffer = ""
        if self._original:
            self._original.flush()


# ---------------------------------------------------------------------------
# Widget (UI)
# ---------------------------------------------------------------------------

class NITorchRegisterWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    _regCounter = 0  # class-level counter for numbering transforms

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = NITorchRegisterLogic()

        # =====================================================================
        # Tab widget
        # =====================================================================
        self.tabWidget = qt.QTabWidget()
        self.layout.addWidget(self.tabWidget)

        # =====================================================================
        # Registration tab
        # =====================================================================
        regWidget = qt.QWidget()
        regLayout = qt.QFormLayout(regWidget)
        self.tabWidget.addTab(regWidget, "Registration")

        # Fixed volume
        self.fixedSelector = slicer.qMRMLNodeComboBox()
        self.fixedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode",
                                        "vtkMRMLLabelMapVolumeNode"]
        self.fixedSelector.selectNodeUponCreation = False
        self.fixedSelector.addEnabled = False
        self.fixedSelector.removeEnabled = False
        self.fixedSelector.noneEnabled = True
        self.fixedSelector.setMRMLScene(slicer.mrmlScene)
        self.fixedSelector.setToolTip("Select the fixed (reference) volume.")
        regLayout.addRow("Fixed:", self.fixedSelector)

        # Moving volume
        self.movingSelector = slicer.qMRMLNodeComboBox()
        self.movingSelector.nodeTypes = ["vtkMRMLScalarVolumeNode",
                                         "vtkMRMLLabelMapVolumeNode"]
        self.movingSelector.selectNodeUponCreation = False
        self.movingSelector.addEnabled = False
        self.movingSelector.removeEnabled = False
        self.movingSelector.noneEnabled = True
        self.movingSelector.setMRMLScene(slicer.mrmlScene)
        self.movingSelector.setToolTip("Select the moving volume.")
        regLayout.addRow("Moving:", self.movingSelector)

        # Categorical checkbox
        self.categoricalCheck = qt.QCheckBox("Categorical (label maps)")
        self.categoricalCheck.checked = False
        self.categoricalCheck.setToolTip(
            "Check this if inputs are label maps. Only Dice loss is available "
            "for categorical data.")
        self.categoricalCheck.toggled.connect(self._onCategoricalToggled)
        regLayout.addRow(self.categoricalCheck)

        # Loss function
        self.lossCombo = qt.QComboBox()
        self._intensityLosses = [("LCC", "lcc"), ("MSE", "mse"), ("NMI", "nmi")]
        self._categoricalLosses = [("Dice", "dice")]
        for label, value in self._intensityLosses:
            self.lossCombo.addItem(label, value)
        self.lossCombo.setToolTip("Similarity metric for registration.")
        regLayout.addRow("Loss Function:", self.lossCombo)

        # Device
        self.deviceCombo = qt.QComboBox()
        self.deviceCombo.addItem("cpu")
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                self.deviceCombo.addItem(f"cuda:{i}")
            if torch.cuda.device_count() > 0:
                self.deviceCombo.setCurrentIndex(1)  # select first CUDA device
        except Exception:
            pass
        self.deviceCombo.setToolTip("Computation device.")
        regLayout.addRow("Device:", self.deviceCombo)

        # Apply transform checkbox
        self.applyTransformCheck = qt.QCheckBox("Apply output transform to moving")
        self.applyTransformCheck.checked = True
        regLayout.addRow(self.applyTransformCheck)

        # Transform selector
        self.transformSelector = slicer.qMRMLNodeComboBox()
        self.transformSelector.nodeTypes = ["vtkMRMLGridTransformNode"]
        self.transformSelector.selectNodeUponCreation = True
        self.transformSelector.addEnabled = False
        self.transformSelector.removeEnabled = False
        self.transformSelector.noneEnabled = True
        self.transformSelector.noneDisplay = "None"
        self.transformSelector.setMRMLScene(slicer.mrmlScene)
        self.transformSelector.setToolTip("Select a grid transform to apply.")
        regLayout.addRow("Output Transform:", self.transformSelector)

        # Register button
        self.registerButton = qt.QPushButton("Run Registration")
        self.registerButton.enabled = False
        regLayout.addRow(self.registerButton)

        # Status label
        self.statusLabel = qt.QLabel("")
        regLayout.addRow(self.statusLabel)

        # --- Advanced section (collapsible) ---
        advancedCollapsible = slicer.qMRMLCollapsibleButton()
        advancedCollapsible.text = "Parameters"
        advancedCollapsible.collapsed = False
        regLayout.addRow(advancedCollapsible)

        advancedLayout = qt.QFormLayout(advancedCollapsible)

        # --- Parameters ordered alphabetically ---

        self.affineBasisCombo = qt.QComboBox()
        self.affineBasisCombo.addItems([
            "similitude", "affine", "rigid", "rotation", "translation"])
        self.affineBasisCombo.setToolTip(
            "Degrees of freedom for the affine transform.")
        advancedLayout.addRow("Affine Basis:", self.affineBasisCombo)

        self.affMaxIterSpin = qt.QSpinBox()
        self.affMaxIterSpin.setRange(1, 1000)
        self.affMaxIterSpin.setValue(128)
        self.affMaxIterSpin.setToolTip("Max iterations for affine optimizer (per pyramid level).")
        advancedLayout.addRow("Affine Max Iter:", self.affMaxIterSpin)

        self.affTolSpin = qt.QDoubleSpinBox()
        self.affTolSpin.setDecimals(4)
        self.affTolSpin.setRange(0.0001, 1.0)
        self.affTolSpin.setSingleStep(0.001)
        self.affTolSpin.setValue(0.0001)
        self.affTolSpin.setToolTip("Convergence tolerance for affine optimizer.")
        advancedLayout.addRow("Affine Tolerance:", self.affTolSpin)

        self.outerMaxIterSpin = qt.QSpinBox()
        self.outerMaxIterSpin.setRange(1, 1000)
        self.outerMaxIterSpin.setValue(64)
        self.outerMaxIterSpin.setToolTip(
            "Max outer iterations for interleaved affine/nonlinear optimization.")
        advancedLayout.addRow("Global Max Iter:", self.outerMaxIterSpin)

        self.outerTolSpin = qt.QDoubleSpinBox()
        self.outerTolSpin.setDecimals(4)
        self.outerTolSpin.setRange(0.0001, 1.0)
        self.outerTolSpin.setSingleStep(0.001)
        self.outerTolSpin.setValue(0.0010)
        self.outerTolSpin.setToolTip(
            "Convergence tolerance for outer interleaved optimization.")
        advancedLayout.addRow("Global Tolerance:", self.outerTolSpin)

        self.regLambdaSpin = qt.QDoubleSpinBox()
        self.regLambdaSpin.setRange(0.01, 1000.0)
        self.regLambdaSpin.setSingleStep(1.0)
        self.regLambdaSpin.setValue(5.0)
        self.regLambdaSpin.setToolTip(
            "Global scaling factor applied to all individual penalty values.")
        advancedLayout.addRow("Lambda Global:", self.regLambdaSpin)

        # --- Individual regularization penalties ---
        self.penAbsoluteSpin = qt.QDoubleSpinBox()
        self.penAbsoluteSpin.setDecimals(4)
        self.penAbsoluteSpin.setRange(0.0, 100.0)
        self.penAbsoluteSpin.setSingleStep(0.0001)
        self.penAbsoluteSpin.setValue(0.0001)
        self.penAbsoluteSpin.setToolTip(
            "Penalty on absolute displacements (0th order).")
        advancedLayout.addRow("Lambda Absolute:", self.penAbsoluteSpin)

        self.penMembraneSpin = qt.QDoubleSpinBox()
        self.penMembraneSpin.setDecimals(4)
        self.penMembraneSpin.setRange(0.0, 100.0)
        self.penMembraneSpin.setSingleStep(0.001)
        self.penMembraneSpin.setValue(0.0010)
        self.penMembraneSpin.setToolTip(
            "Penalty on membrane energy (1st order).")
        advancedLayout.addRow("Lambda Membrane:", self.penMembraneSpin)

        self.penBendingSpin = qt.QDoubleSpinBox()
        self.penBendingSpin.setDecimals(4)
        self.penBendingSpin.setRange(0.0, 100.0)
        self.penBendingSpin.setSingleStep(0.01)
        self.penBendingSpin.setValue(0.2000)
        self.penBendingSpin.setToolTip(
            "Penalty on bending energy (2nd order).")
        advancedLayout.addRow("Lambda Bending:", self.penBendingSpin)

        self.penLameEdit = qt.QLineEdit("0.05, 0.2")
        self.penLameEdit.setToolTip(
            "Lame constants for linear elastic energy (two comma-separated values).")
        advancedLayout.addRow("Lambda Lame:", self.penLameEdit)

        self.nlMaxIterSpin = qt.QSpinBox()
        self.nlMaxIterSpin.setRange(1, 1000)
        self.nlMaxIterSpin.setValue(64)
        self.nlMaxIterSpin.setToolTip("Max iterations for nonlinear optimizer (per pyramid level).")
        advancedLayout.addRow("Nonlin Max Iter:", self.nlMaxIterSpin)

        self.nlTolSpin = qt.QDoubleSpinBox()
        self.nlTolSpin.setDecimals(4)
        self.nlTolSpin.setRange(0.0001, 1.0)
        self.nlTolSpin.setSingleStep(0.001)
        self.nlTolSpin.setValue(0.0010)
        self.nlTolSpin.setToolTip("Convergence tolerance for nonlinear optimizer.")
        advancedLayout.addRow("Nonlin Tolerance:", self.nlTolSpin)

        self.pyramidLevelsEdit = qt.QLineEdit("0, 1, 2")
        self.pyramidLevelsEdit.setToolTip(
            "Comma-separated pyramid levels (0=finest). "
            "More levels = coarser initial alignment.")
        advancedLayout.addRow("Pyramid Levels:", self.pyramidLevelsEdit)

        resetDefaultsButton = qt.QPushButton("Reset to Defaults")
        resetDefaultsButton.clicked.connect(self._resetParameterDefaults)
        advancedLayout.addRow(resetDefaultsButton)

        # --- Log section (collapsible, expanded) ---
        logCollapsible = slicer.qMRMLCollapsibleButton()
        logCollapsible.text = "Log"
        logCollapsible.collapsed = False
        regLayout.addRow(logCollapsible)

        logLayout = qt.QVBoxLayout(logCollapsible)

        self.logText = qt.QPlainTextEdit()
        self.logText.readOnly = True
        self.logText.setMinimumHeight(200)
        font = qt.QFont("Monospace")
        font.setStyleHint(qt.QFont.TypeWriter)
        self.logText.setFont(font)
        logLayout.addWidget(self.logText)

        clearLogButton = qt.QPushButton("Clear Log")
        clearLogButton.clicked.connect(lambda: self.logText.clear())
        logLayout.addWidget(clearLogButton)

        # --- Visualization section (collapsible, collapsed) ---
        vizCollapsible = slicer.qMRMLCollapsibleButton()
        vizCollapsible.text = "Visualization"
        vizCollapsible.collapsed = False
        regLayout.addRow(vizCollapsible)

        vizLayout = qt.QFormLayout(vizCollapsible)

        vizEnableRow = qt.QHBoxLayout()
        self.vizEnableCheck = qt.QCheckBox("Enable comparison view")
        self.vizEnableCheck.checked = False
        self.vizEnableCheck.enabled = False
        self.vizEnableCheck.setToolTip(
            "Overlay fixed and moving volumes in slice views. "
            "Use the opacity slider to fade between them.")
        vizEnableRow.addWidget(self.vizEnableCheck)
        self.vizContoursCheck = qt.QCheckBox("Contours")
        self.vizContoursCheck.checked = False
        self.vizContoursCheck.enabled = False
        self.vizContoursCheck.setToolTip(
            "Overlay edge contours of the moving image on the fixed image.")
        vizEnableRow.addWidget(self.vizContoursCheck)
        vizEnableRow.addStretch()
        vizLayout.addRow(vizEnableRow)

        self.vizOpacitySlider = qt.QSlider(qt.Qt.Horizontal)
        self.vizOpacitySlider.enabled = False
        self.vizOpacitySlider.minimum = 0
        self.vizOpacitySlider.maximum = 100
        self.vizOpacitySlider.value = 50
        self.vizOpacitySlider.setToolTip(
            "0 = show fixed only, 100 = show moving only.")
        vizLayout.addRow("Opacity:", self.vizOpacitySlider)

        self.vizContourDensitySlider = qt.QSlider(qt.Qt.Horizontal)
        self.vizContourDensitySlider.enabled = False
        self.vizContourDensitySlider.minimum = 1
        self.vizContourDensitySlider.maximum = 100
        self.vizContourDensitySlider.value = 12
        self.vizContourDensitySlider.setToolTip(
            "Contour density: lower = fewer/cleaner, higher = more/finer.")
        vizLayout.addRow("Contour Density:", self.vizContourDensitySlider)

        # State for save/restore
        self._vizSavedState = None
        self._contourNode = None
        self._contourColorNode = None

        # Add spacer at end of registration tab
        regLayout.addRow(qt.QWidget())  # spacer

        # =====================================================================
        # Validation tab
        # =====================================================================
        valWidget = qt.QWidget()
        valLayout = qt.QFormLayout(valWidget)
        self.tabWidget.addTab(valWidget, "Validation")

        # Reference volume (fixed image) selector
        self.refVolumeSelector = slicer.qMRMLNodeComboBox()
        self.refVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode",
                                             "vtkMRMLLabelMapVolumeNode"]
        self.refVolumeSelector.selectNodeUponCreation = False
        self.refVolumeSelector.addEnabled = False
        self.refVolumeSelector.removeEnabled = False
        self.refVolumeSelector.noneEnabled = True
        self.refVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.refVolumeSelector.setToolTip(
            "Reference volume for label export geometry (typically the fixed image).")
        valLayout.addRow("Fixed:", self.refVolumeSelector)

        # Fixed labels selector
        self.fixedLabelsSelector = slicer.qMRMLNodeComboBox()
        self.fixedLabelsSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.fixedLabelsSelector.selectNodeUponCreation = False
        self.fixedLabelsSelector.addEnabled = False
        self.fixedLabelsSelector.removeEnabled = False
        self.fixedLabelsSelector.noneEnabled = True
        self.fixedLabelsSelector.setMRMLScene(slicer.mrmlScene)
        self.fixedLabelsSelector.setToolTip(
            "Select the fixed (reference) segmentation.")
        valLayout.addRow("Fixed Labels:", self.fixedLabelsSelector)

        # Moving labels selector
        self.movingLabelsSelector = slicer.qMRMLNodeComboBox()
        self.movingLabelsSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.movingLabelsSelector.selectNodeUponCreation = False
        self.movingLabelsSelector.addEnabled = False
        self.movingLabelsSelector.removeEnabled = False
        self.movingLabelsSelector.noneEnabled = True
        self.movingLabelsSelector.setMRMLScene(slicer.mrmlScene)
        self.movingLabelsSelector.setToolTip(
            "Select the moving segmentation.")
        valLayout.addRow("Moving Labels:", self.movingLabelsSelector)

        # Transform selector for validation
        self.valTransformSelector = slicer.qMRMLNodeComboBox()
        self.valTransformSelector.nodeTypes = ["vtkMRMLGridTransformNode"]
        self.valTransformSelector.selectNodeUponCreation = False
        self.valTransformSelector.addEnabled = False
        self.valTransformSelector.removeEnabled = False
        self.valTransformSelector.noneEnabled = True
        self.valTransformSelector.noneDisplay = "None"
        self.valTransformSelector.setMRMLScene(slicer.mrmlScene)
        self.valTransformSelector.setToolTip(
            "Select a grid transform to evaluate. "
            "Dice is computed both without and with this transform.")
        valLayout.addRow("Output Transform:", self.valTransformSelector)

        # Compute button
        self.computeDiceButton = qt.QPushButton("Compute Dice")
        self.computeDiceButton.enabled = False
        valLayout.addRow(self.computeDiceButton)

        # Summary table (mean Dice per run)
        summaryLabel = qt.QLabel("Summary (mean Dice per run):")
        valLayout.addRow(summaryLabel)

        self.summaryTable = qt.QTableWidget()
        self.summaryTable.setMinimumHeight(80)
        self.summaryTable.setMaximumHeight(150)
        self.summaryTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.summaryTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.summaryTable.horizontalHeader().setSectionResizeMode(
            qt.QHeaderView.Stretch)
        valLayout.addRow(self.summaryTable)

        # Result selector
        self.diceResultSelector = qt.QComboBox()
        self.diceResultSelector.setToolTip(
            "Select a previous Dice result to view per-label scores.")
        valLayout.addRow("Result:", self.diceResultSelector)

        # Mean Dice display for selected result
        self.diceMeanLabel = qt.QLabel("")
        font = qt.QFont()
        font.setBold(True)
        self.diceMeanLabel.setFont(font)
        valLayout.addRow("Mean Dice:", self.diceMeanLabel)

        # Per-label results table
        self.diceTable = qt.QTableWidget()
        self.diceTable.setMinimumHeight(250)
        self.diceTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.diceTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.diceTable.horizontalHeader().setSectionResizeMode(
            qt.QHeaderView.Stretch)
        valLayout.addRow(self.diceTable)

        # Clear results button
        clearDiceButton = qt.QPushButton("Clear Results")
        clearDiceButton.clicked.connect(self._onClearDice)
        valLayout.addRow(clearDiceButton)

        # Store dice results per transform for comparison
        self._diceResults = {}  # {transformName: {per_label, mean_before, mean_after}}

        # --- Connections ---
        self.fixedSelector.currentNodeChanged.connect(self._updateRegButtonState)
        self.movingSelector.currentNodeChanged.connect(self._updateRegButtonState)
        self.registerButton.clicked.connect(self._onRegister)

        self.applyTransformCheck.toggled.connect(self._onApplyTransformToggled)
        self.transformSelector.currentNodeChanged.connect(self._onTransformSelected)
        self.vizEnableCheck.toggled.connect(self._onVizToggled)
        self.vizOpacitySlider.valueChanged.connect(self._onVizOpacityChanged)
        self.vizContoursCheck.toggled.connect(self._onContoursToggled)
        self.vizContourDensitySlider.sliderReleased.connect(self._onContourDensityChanged)

        self.fixedLabelsSelector.currentNodeChanged.connect(self._updateDiceButtonState)
        self.movingLabelsSelector.currentNodeChanged.connect(self._updateDiceButtonState)
        self.refVolumeSelector.currentNodeChanged.connect(self._updateDiceButtonState)
        self.computeDiceButton.clicked.connect(self._onComputeDice)
        self.diceResultSelector.currentIndexChanged.connect(self._onDiceResultSelected)

    # -----------------------------------------------------------------
    # UI helpers
    # -----------------------------------------------------------------

    def _onCategoricalToggled(self, checked):
        self.lossCombo.clear()
        items = self._categoricalLosses if checked else self._intensityLosses
        for label, value in items:
            self.lossCombo.addItem(label, value)

    def _updateRegButtonState(self, _=None):
        hasBoth = (self.fixedSelector.currentNode() is not None
                   and self.movingSelector.currentNode() is not None)
        self.registerButton.enabled = hasBoth
        self.vizEnableCheck.enabled = hasBoth
        self.vizContoursCheck.enabled = hasBoth
        self.vizOpacitySlider.enabled = hasBoth
        self.vizContourDensitySlider.enabled = hasBoth and self.vizContoursCheck.checked

    def _resetParameterDefaults(self):
        self.affineBasisCombo.setCurrentIndex(0)  # similitude
        self.affMaxIterSpin.setValue(128)
        self.affTolSpin.setValue(0.0001)
        self.outerMaxIterSpin.setValue(64)
        self.outerTolSpin.setValue(0.0010)
        self.regLambdaSpin.setValue(5.0)
        self.penAbsoluteSpin.setValue(0.0001)
        self.penMembraneSpin.setValue(0.0010)
        self.penBendingSpin.setValue(0.2000)
        self.penLameEdit.setText("0.05, 0.2")
        self.nlMaxIterSpin.setValue(64)
        self.nlTolSpin.setValue(0.0010)
        self.pyramidLevelsEdit.setText("0, 1, 2")

    def _applyCurrentTransform(self):
        """Apply or remove the selected transform based on checkbox state."""
        movingNode = self.movingSelector.currentNode()
        if movingNode is None:
            return
        transformNode = self.transformSelector.currentNode()
        if self.applyTransformCheck.checked and transformNode:
            movingNode.SetAndObserveTransformNodeID(transformNode.GetID())
        else:
            movingNode.SetAndObserveTransformNodeID(None)
        self._syncContourTransform()

    def _syncContourTransform(self):
        """Sync the contour node's transform with the moving volume's."""
        if self._contourNode is None:
            return
        movingNode = self.movingSelector.currentNode()
        if movingNode is None:
            return
        self._contourNode.SetAndObserveTransformNodeID(
            movingNode.GetTransformNodeID())

    def _onApplyTransformToggled(self, checked):
        self._applyCurrentTransform()

    def _onTransformSelected(self, node):
        self._applyCurrentTransform()

    # -----------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------

    def _saveSliceViewState(self):
        """Save the current slice view state (volumes, opacity, compositing)."""
        layoutManager = slicer.app.layoutManager()
        state = {"layout": layoutManager.layout, "views": {}}
        for name in layoutManager.sliceViewNames():
            sliceWidget = layoutManager.sliceWidget(name)
            compositeNode = sliceWidget.sliceLogic().GetSliceCompositeNode()
            sliceNode = sliceWidget.sliceLogic().GetSliceNode()
            state["views"][name] = {
                "bg": compositeNode.GetBackgroundVolumeID(),
                "fg": compositeNode.GetForegroundVolumeID(),
                "label": compositeNode.GetLabelVolumeID(),
                "opacity": compositeNode.GetForegroundOpacity(),
                "compositing": compositeNode.GetCompositing(),
                "labelOutline": sliceNode.GetUseLabelOutline(),
                "linked": compositeNode.GetLinkedControl(),
            }
        # Save crosshair state
        crosshairNode = slicer.util.getNode("Crosshair")
        state["crosshairMode"] = crosshairNode.GetCrosshairMode()
        return state

    def _restoreSliceViewState(self, state):
        """Restore a previously saved slice view state."""
        if state is None:
            return
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(state["layout"])
        slicer.app.processEvents()
        for name in layoutManager.sliceViewNames():
            if name not in state["views"]:
                continue
            info = state["views"][name]
            sliceWidget = layoutManager.sliceWidget(name)
            compositeNode = sliceWidget.sliceLogic().GetSliceCompositeNode()
            sliceNode = sliceWidget.sliceLogic().GetSliceNode()
            compositeNode.SetBackgroundVolumeID(info["bg"])
            compositeNode.SetForegroundVolumeID(info["fg"])
            compositeNode.SetLabelVolumeID(info["label"])
            compositeNode.SetForegroundOpacity(info["opacity"])
            compositeNode.SetCompositing(info["compositing"])
            sliceNode.SetUseLabelOutline(info["labelOutline"])
            compositeNode.SetLinkedControl(info["linked"])
        # Restore crosshair state
        crosshairNode = slicer.util.getNode("Crosshair")
        crosshairNode.SetCrosshairMode(state.get("crosshairMode", 0))

    def _applyVizMode(self, fit=False):
        """Apply fade visualization to slice views."""
        fixedNode = self.fixedSelector.currentNode()
        movingNode = self.movingSelector.currentNode()
        if fixedNode is None or movingNode is None:
            return

        opacity = self.vizOpacitySlider.value / 100.0
        fixedID = fixedNode.GetID()
        movingID = movingNode.GetID()

        layoutManager = slicer.app.layoutManager()
        for name in layoutManager.sliceViewNames():
            compositeNode = layoutManager.sliceWidget(
                name).sliceLogic().GetSliceCompositeNode()
            compositeNode.SetBackgroundVolumeID(fixedID)
            compositeNode.SetForegroundVolumeID(movingID)
            compositeNode.SetForegroundOpacity(opacity)
            compositeNode.SetCompositing(0)
            compositeNode.SetLinkedControl(True)

        if fit:
            for name in layoutManager.sliceViewNames():
                layoutManager.sliceWidget(name).sliceLogic().FitSliceToAll()

        # Enable crosshair
        crosshairNode = slicer.util.getNode("Crosshair")
        crosshairNode.SetCrosshairMode(
            slicer.vtkMRMLCrosshairNode.ShowBasic)

    def _createContourNode(self, movingNode):
        """Create a label map of iso-contours from the moving volume."""
        import numpy as np
        from scipy.ndimage import binary_erosion

        data = slicer.util.arrayFromVolume(movingNode).astype(np.float64)

        # Normalize intensity to [0, 1] using robust percentiles
        nonzero = data[data > 0]
        if nonzero.size == 0:
            return None
        dmin, dmax = np.percentile(nonzero, [2, 98])
        if dmax <= dmin:
            return None
        normalized = np.clip((data - dmin) / (dmax - dmin), 0, 1)

        # Number of iso-levels from density slider (1->2, 100->8)
        density = self.vizContourDensitySlider.value
        n_levels = int(2 + (density - 1) * 6 / 99)
        levels = np.linspace(0.15, 0.85, n_levels)

        # Build iso-contours: boundary of thresholded regions at each level
        edges = np.zeros(data.shape, dtype=bool)
        for level in levels:
            mask = normalized > level
            eroded = binary_erosion(mask)
            edges |= (mask ^ eroded)

        edges = edges.astype(np.int16)

        # Create label map node
        contourNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "_NITorch_contours")
        contourNode.SetHideFromEditors(True)

        # Copy geometry from moving volume
        import vtk
        mat = vtk.vtkMatrix4x4()
        movingNode.GetIJKToRASMatrix(mat)
        contourNode.SetIJKToRASMatrix(mat)

        slicer.util.updateVolumeFromArray(contourNode, edges)

        # Ensure display node exists
        contourNode.CreateDefaultDisplayNodes()

        # Set contour color to red
        colorNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLColorTableNode", "_NITorch_contour_color")
        colorNode.SetHideFromEditors(True)
        colorNode.SetTypeToUser()
        colorNode.SetNumberOfColors(2)
        colorNode.SetColor(0, "Background", 0.0, 0.0, 0.0, 0.0)
        colorNode.SetColor(1, "Contour", 1.0, 0.0, 0.0, 1.0)
        contourNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())
        self._contourColorNode = colorNode

        # Apply the same transform as the moving volume
        transformID = movingNode.GetTransformNodeID()
        if transformID:
            contourNode.SetAndObserveTransformNodeID(transformID)

        return contourNode

    def _applyContours(self):
        """Show or hide contour overlay based on checkbox state."""
        if not self.vizContoursCheck.checked:
            self._removeContourNode()
            # Clear label layer
            slicer.util.setSliceViewerLayers(label=None)
            return

        movingNode = self.movingSelector.currentNode()
        if movingNode is None:
            return

        # Remove old contour node before creating new one
        self._removeContourNode()
        self._contourNode = self._createContourNode(movingNode)
        if self._contourNode is None:
            return

        # Set as label layer with outline display
        slicer.util.setSliceViewerLayers(label=self._contourNode)
        layoutManager = slicer.app.layoutManager()
        for name in layoutManager.sliceViewNames():
            sliceNode = layoutManager.sliceWidget(
                name).sliceLogic().GetSliceNode()
            sliceNode.SetUseLabelOutline(True)

    def _removeContourNode(self):
        """Remove the contour label map and color nodes from the scene."""
        if self._contourNode is not None:
            slicer.mrmlScene.RemoveNode(self._contourNode)
            self._contourNode = None
        if self._contourColorNode is not None:
            slicer.mrmlScene.RemoveNode(self._contourColorNode)
            self._contourColorNode = None

    def _onVizToggled(self, checked):
        if checked:
            # Save current state before changing views
            self._vizSavedState = self._saveSliceViewState()
            self._applyVizMode(fit=True)
            self._applyContours()
        else:
            self._removeContourNode()
            self._restoreSliceViewState(self._vizSavedState)
            self._vizSavedState = None

    def _onVizOpacityChanged(self, value):
        if not self.vizEnableCheck.checked:
            return
        opacity = value / 100.0
        for compositeNode in slicer.util.getNodesByClass(
                "vtkMRMLSliceCompositeNode"):
            compositeNode.SetForegroundOpacity(opacity)

    def _onContoursToggled(self, checked):
        self.vizContourDensitySlider.enabled = checked
        if self.vizEnableCheck.checked:
            self._applyContours()

    def _onContourDensityChanged(self):
        if self.vizEnableCheck.checked and self.vizContoursCheck.checked:
            self._applyContours()

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _updateDiceButtonState(self, _=None):
        self.computeDiceButton.enabled = (
            self.fixedLabelsSelector.currentNode() is not None
            and self.movingLabelsSelector.currentNode() is not None
            and self.refVolumeSelector.currentNode() is not None
        )

    def _segmentationToLabelArray(self, segNode, referenceVolumeNode):
        """Export a segmentation node to a numpy label array in reference geometry.

        Parameters
        ----------
        segNode : vtkMRMLSegmentationNode
            Segmentation to export.
        referenceVolumeNode : vtkMRMLVolumeNode
            Reference volume node (for geometry).

        Returns
        -------
        np.ndarray
            3D integer label array.
        """
        import numpy as np
        import vtk

        # Create a temporary label map in reference geometry
        labelmapNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "_NITorch_temp_labelmap")
        try:
            segIDs = vtk.vtkStringArray()
            segNode.GetSegmentation().GetSegmentIDs(segIDs)
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segNode, segIDs, labelmapNode, referenceVolumeNode)
            arr = slicer.util.arrayFromVolume(labelmapNode).copy()
        finally:
            slicer.mrmlScene.RemoveNode(labelmapNode)
        return arr

    def _onComputeDice(self):
        import numpy as np
        from NITorchRegisterLib.validation import compute_dice_scores

        fixedSegNode = self.fixedLabelsSelector.currentNode()
        movingSegNode = self.movingLabelsSelector.currentNode()
        refVolumeNode = self.refVolumeSelector.currentNode()
        if fixedSegNode is None or movingSegNode is None or refVolumeNode is None:
            slicer.util.errorDisplay(
                "Please select fixed labels, moving labels, and a reference volume.")
            return

        transformNode = self.valTransformSelector.currentNode()
        transformName = transformNode.GetName() if transformNode else "(none)"

        # Export segmentations to label arrays in reference geometry
        fixed_labels = self._segmentationToLabelArray(
            fixedSegNode, refVolumeNode)

        # For "before" Dice: moving labels without any transform
        moving_labels_before = self._segmentationToLabelArray(
            movingSegNode, refVolumeNode)

        # For "after" Dice: apply the selected transform to moving, then export
        if transformNode:
            # Clone segmentation, apply transform, export, then remove clone
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            itemID = shNode.GetItemByDataNode(movingSegNode)
            cloneItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(
                shNode, itemID)
            cloneNode = shNode.GetItemDataNode(cloneItemID)
            try:
                cloneNode.SetAndObserveTransformNodeID(transformNode.GetID())
                cloneNode.HardenTransform()
                moving_labels_after = self._segmentationToLabelArray(
                    cloneNode, refVolumeNode)
            finally:
                slicer.mrmlScene.RemoveNode(cloneNode)
        else:
            moving_labels_after = moving_labels_before

        # Compute Dice scores
        dice_before, mean_before = compute_dice_scores(
            fixed_labels, moving_labels_before)
        dice_after, mean_after = compute_dice_scores(
            fixed_labels, moving_labels_after)

        # Store results
        self._diceResults[transformName] = {
            "per_label": {},
            "mean_before": mean_before,
            "mean_after": mean_after,
        }
        all_labels = sorted(set(dice_before.keys()) | set(dice_after.keys()))
        for lab in all_labels:
            self._diceResults[transformName]["per_label"][lab] = (
                dice_before.get(lab, 0.0), dice_after.get(lab, 0.0))

        # Update selector and tables
        self._updateDiceSelector(transformName)
        self._updateSummaryTable()

    def _onClearDice(self):
        self._diceResults = {}
        self.diceResultSelector.clear()
        self.diceMeanLabel.text = ""
        self.diceTable.setRowCount(0)
        self.diceTable.setColumnCount(0)
        self.summaryTable.setRowCount(0)
        self.summaryTable.setColumnCount(0)

    def _updateDiceSelector(self, selectName=None):
        """Update the result selector dropdown."""
        self.diceResultSelector.blockSignals(True)
        self.diceResultSelector.clear()
        for name in self._diceResults:
            self.diceResultSelector.addItem(name)
        if selectName and selectName in self._diceResults:
            idx = list(self._diceResults.keys()).index(selectName)
            self.diceResultSelector.setCurrentIndex(idx)
        self.diceResultSelector.blockSignals(False)
        self._updateDiceTable()

    def _onDiceResultSelected(self, _index):
        self._updateDiceTable()

    def _updateSummaryTable(self):
        """Update the summary table showing mean Dice for all runs."""
        transforms = list(self._diceResults.keys())
        self.summaryTable.setRowCount(len(transforms))
        self.summaryTable.setColumnCount(3)
        self.summaryTable.setHorizontalHeaderLabels(
            ["Transform", "Before", "After"])

        font = qt.QFont()
        font.setBold(True)

        for row, t in enumerate(transforms):
            self.summaryTable.setItem(row, 0, qt.QTableWidgetItem(t))
            mb = self._diceResults[t]["mean_before"]
            ma = self._diceResults[t]["mean_after"]
            self.summaryTable.setItem(
                row, 1, qt.QTableWidgetItem(f"{mb:.4f}"))
            item_a = qt.QTableWidgetItem(f"{ma:.4f}")
            if ma > mb + 0.001:
                item_a.setBackground(qt.QColor(200, 255, 200))
            elif ma < mb - 0.001:
                item_a.setBackground(qt.QColor(255, 200, 200))
            self.summaryTable.setItem(row, 2, item_a)


    def _updateDiceTable(self):
        """Update the mean label and per-label Dice table for the selected result."""
        name = self.diceResultSelector.currentText
        if not name or name not in self._diceResults:
            self.diceMeanLabel.text = ""
            self.diceTable.setRowCount(0)
            self.diceTable.setColumnCount(0)
            return

        result = self._diceResults[name]
        mb = result["mean_before"]
        ma = result["mean_after"]
        self.diceMeanLabel.text = f"{mb:.4f} \u2192 {ma:.4f}"

        all_labels = sorted(result["per_label"].keys())

        self.diceTable.setRowCount(len(all_labels))
        self.diceTable.setColumnCount(3)
        self.diceTable.setHorizontalHeaderLabels(["Label", "Before", "After"])

        for row, lab in enumerate(all_labels):
            before, after = result["per_label"][lab]
            self.diceTable.setItem(row, 0, qt.QTableWidgetItem(str(lab)))
            self.diceTable.setItem(
                row, 1, qt.QTableWidgetItem(f"{before:.4f}"))
            item_a = qt.QTableWidgetItem(f"{after:.4f}")
            if after > before + 0.001:
                item_a.setBackground(qt.QColor(200, 255, 200))
            elif after < before - 0.001:
                item_a.setBackground(qt.QColor(255, 200, 200))
            self.diceTable.setItem(row, 2, item_a)


    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------

    def _onRegister(self):
        fixedNode = self.fixedSelector.currentNode()
        movingNode = self.movingSelector.currentNode()
        if fixedNode is None or movingNode is None:
            slicer.util.errorDisplay("Please select both fixed and moving volumes.")
            return

        # Validate 3D inputs
        for name, node in [("Fixed", fixedNode), ("Moving", movingNode)]:
            dims = [node.GetImageData().GetDimensions()[i] for i in range(3)]
            if any(d <= 1 for d in dims):
                slicer.util.errorDisplay(
                    f"{name} volume is not 3D (dimensions: {dims[0]}x{dims[1]}x{dims[2]}). "
                    "Please select a 3D volume.")
                return

        # Parse pyramid levels
        try:
            pyramid_levels = [
                int(x.strip())
                for x in self.pyramidLevelsEdit.text.split(",")
                if x.strip()
            ]
        except ValueError:
            slicer.util.errorDisplay("Invalid pyramid levels. Use comma-separated integers.")
            return

        # Parse Lame constants
        try:
            lame_vals = [
                float(x.strip())
                for x in self.penLameEdit.text.split(",")
                if x.strip()
            ]
            if len(lame_vals) != 2:
                raise ValueError
            penalty_lame = tuple(lame_vals)
        except ValueError:
            slicer.util.errorDisplay(
                "Invalid Lambda Lame. Use two comma-separated values (e.g. 0.05, 0.2).")
            return

        params = {
            "loss_name": str(self.lossCombo.currentData),
            "device": str(self.deviceCombo.currentText),
            "is_label": bool(self.categoricalCheck.checked),
            "affine_basis": str(self.affineBasisCombo.currentText),
            "pyramid_levels": pyramid_levels,
            "reg_lambda": self.regLambdaSpin.value,
            "penalty_absolute": self.penAbsoluteSpin.value,
            "penalty_membrane": self.penMembraneSpin.value,
            "penalty_bending": self.penBendingSpin.value,
            "penalty_lame": penalty_lame,
            "aff_max_iter": int(self.affMaxIterSpin.value),
            "aff_tolerance": self.affTolSpin.value,
            "nl_max_iter": int(self.nlMaxIterSpin.value),
            "nl_tolerance": self.nlTolSpin.value,
            "outer_max_iter": int(self.outerMaxIterSpin.value),
            "outer_tolerance": self.outerTolSpin.value,
            "apply_transform": bool(self.applyTransformCheck.checked),
        }

        self.registerButton.enabled = False
        self.statusLabel.text = "Running registration..."
        slicer.app.processEvents()

        try:
            NITorchRegisterWidget._regCounter += 1
            counter = NITorchRegisterWidget._regCounter
            elapsed, transformNode = self.logic.runRegistration(
                fixedNode, movingNode, params, self.logText, counter)
            # Select new transform in combo (triggers _onTransformSelected
            # which applies it if checkbox is checked)
            self.transformSelector.setCurrentNode(transformNode)
            self.statusLabel.text = "Registration complete."
        except Exception as e:
            self.statusLabel.text = f"Error: {e}"
            slicer.util.errorDisplay(f"Registration failed:\n{e}")
            logging.exception("Registration failed")
        finally:
            self.registerButton.enabled = True

    # -----------------------------------------------------------------
    # Validation


# ---------------------------------------------------------------------------
# Logic (processing)
# ---------------------------------------------------------------------------

class NITorchRegisterLogic(ScriptedLoadableModuleLogic):

    def runRegistration(self, fixedNode, movingNode, params, logWidget,
                        counter=1):
        """Run the full registration pipeline (blocking, main thread).

        Parameters
        ----------
        fixedNode : vtkMRMLVolumeNode
            Fixed (reference) volume node.
        movingNode : vtkMRMLVolumeNode
            Moving volume node.
        params : dict
            Registration parameters.
        logWidget : QPlainTextEdit
            Log output widget.
        counter : int
            Registration run number for naming transforms.

        Returns
        -------
        elapsed : float
            Total time in seconds.
        transformNode : vtkMRMLGridTransformNode
            The loaded grid transform node.
        """
        import torch
        import numpy as np
        from NITorchRegisterLib.registration import register
        from NITorchRegisterLib.grid_transform import (
            compose_deformation_grid,
            grid_to_slicer_displacement,
        )

        loss_name = params["loss_name"]
        device = params["device"]
        is_label = params["is_label"]

        t_start = time.time()
        transformNode = None

        # Redirect stdout to log widget
        capture = _LogCapture(logWidget, sys.stdout)
        old_stdout = sys.stdout
        sys.stdout = capture

        try:
            # Extract data from Slicer nodes (no disk I/O)
            print("Preparing volumes...")

            # Slicer arrayFromVolume returns (K, J, I) — transpose to (I, J, K)
            fixed_np = np.transpose(
                slicer.util.arrayFromVolume(fixedNode).astype(np.float64))
            moving_np = np.transpose(
                slicer.util.arrayFromVolume(movingNode).astype(np.float64))

            fixed_dat = torch.from_numpy(fixed_np)
            moving_dat = torch.from_numpy(moving_np)

            # Get IJK-to-RAS affine matrices
            fixed_affine = self._getIJKToRASMatrix(fixedNode)
            moving_affine = self._getIJKToRASMatrix(movingNode)
            fixed_shape = tuple(fixed_dat.shape[:3])

            # Run registration
            print(f"Starting registration (device={device}, loss={loss_name})...\n")
            affine_sqrt, displacement, disp_affine = register(
                fixed_dat, fixed_affine, moving_dat, moving_affine,
                loss_name,
                is_label=is_label,
                affine_basis=params["affine_basis"],
                reg_lambda=params["reg_lambda"],
                penalty_absolute=params["penalty_absolute"],
                penalty_membrane=params["penalty_membrane"],
                penalty_bending=params["penalty_bending"],
                penalty_lame=params["penalty_lame"],
                aff_max_iter=params["aff_max_iter"],
                aff_tolerance=params["aff_tolerance"],
                nl_max_iter=params["nl_max_iter"],
                nl_tolerance=params["nl_tolerance"],
                outer_max_iter=params["outer_max_iter"],
                outer_tolerance=params["outer_tolerance"],
                pyramid_levels=params.get("pyramid_levels"),
                device=device,
                verbose=True,
            )

            # Compose deformation grid
            print("\nComposing deformation grid...")
            slicer.app.processEvents()
            grid = compose_deformation_grid(
                fixed_shape, fixed_affine, affine_sqrt,
                displacement, disp_affine, device=device,
            )

            # Create grid transform
            print("Creating grid transform...")
            slicer.app.processEvents()
            disp_np = grid_to_slicer_displacement(
                grid, fixed_shape, fixed_affine, device=device)
            fixed_affine_np = fixed_affine.cpu().numpy().astype(np.float64)

            transformNode = self._createGridTransformNode(
                disp_np, fixed_affine_np, fixed_shape)
            transformNode.SetName(
                f"NITorch_{counter:03d}_{params['loss_name']}_"
                f"{fixedNode.GetName()}_to_{movingNode.GetName()}")

            elapsed = time.time() - t_start
            print(f"\nRegistration complete in {elapsed:.1f}s "
                  f"({elapsed / 60:.1f}min).")

        finally:
            capture.flush()
            sys.stdout = old_stdout

        return elapsed, transformNode

    @staticmethod
    def _getIJKToRASMatrix(volumeNode):
        """Extract the IJK-to-RAS matrix from a Slicer volume node as a torch tensor."""
        import vtk
        import torch
        import numpy as np
        mat = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(mat)
        arr = np.zeros((4, 4), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                arr[i, j] = mat.GetElement(i, j)
        return torch.from_numpy(arr)

    @staticmethod
    def _createGridTransformNode(disp_np, affine_np, fixed_shape):
        """Create a grid transform node from a displacement array.

        Writes a transient temp NIfTI (the only disk I/O in the pipeline),
        loads it via slicer.util.loadTransform which handles all coordinate
        conventions correctly, then deletes the file immediately.

        Parameters
        ----------
        disp_np : np.ndarray
            (X, Y, Z, 1, 3) displacement field in RAS.
        affine_np : np.ndarray
            (4, 4) IJK-to-RAS affine of the fixed image.
        fixed_shape : tuple
            (X, Y, Z) shape.

        Returns
        -------
        transformNode : vtkMRMLGridTransformNode
        """
        import tempfile
        import nibabel as nib

        tmpdir = tempfile.mkdtemp(prefix="nitorch_grid_")
        grid_path = os.path.join(tmpdir, "Grid.nii.gz")
        try:
            grid_nii = nib.Nifti1Image(disp_np, affine_np)
            grid_nii.header['intent_code'] = 1006  # NIFTI_INTENT_DISPVECT
            nib.save(grid_nii, grid_path)
            transformNode = slicer.util.loadTransform(grid_path)
        finally:
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except OSError:
                pass

        return transformNode
