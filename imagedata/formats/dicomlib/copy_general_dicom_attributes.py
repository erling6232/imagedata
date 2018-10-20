#!/usr/bin/env python

# Copyright (c) 2013 Erling Andersen, Haukeland University Hospital


def copy_general_dicom_attributes(ds_in, ds_out):
	# Patient Module Attributes
	ds_out.PatientName               = ds_in.PatientName
	ds_out.PatientID                 = ds_in.PatientID
	if 'PatientBirthDate' in ds_in: ds_out.PatientBirthDate = ds_in.PatientBirthDate
	if 'PatientSex' in ds_in: ds_out.PatientSex = ds_in.PatientSex
	# General Study Module Attributes
	ds_out.StudyInstanceUID          = ds_in.StudyInstanceUID
	if 'StudyDate' in ds_in: ds_out.StudyDate = ds_in.StudyDate
	if 'StudyTime' in ds_in: ds_out.StudyTime = ds_in.StudyTime
	if 'ReferringPhysicianName' in ds_in: ds_out.ReferringPhysicianName = ds_in.ReferringPhysicianName
	if 'StudyID' in ds_in: ds_out.StudyID = ds_in.StudyID
	if 'AccessionNumber' in ds_in: ds_out.AccessionNumber = ds_in.AccessionNumber
	if 'StudyDescription' in ds_in: ds_out.StudyDescription = ds_in.StudyDescription
	# Patient Study Module Attributes
	if 'AdmittingDiagnosesDescription' in ds_in: ds_out.AdmittingDiagnosesDescription = ds_in.AdmittingDiagnosesDescription
	if 'PatientAge' in ds_in: ds_out.PatientAge = ds_in.PatientAge
	if 'PatientSize' in ds_in: ds_out.PatientSize = ds_in.PatientSize
	if 'PatientWeight' in ds_in: ds_out.PatientWeight = ds_in.PatientWeight
	# General Series Module Attributes
	ds_out.Modality                  = ds_in.Modality
	ds_out.SeriesInstanceUID         = ds_in.SeriesInstanceUID
	if 'SeriesNumber' in ds_in: ds_out.SeriesNumber = ds_in.SeriesNumber
	if 'Laterality' in ds_in: ds_out.Laterality = ds_in.Laterality
	if 'SeriesDate' in ds_in: ds_out.SeriesDate = ds_in.SeriesDate
	if 'SeriesTime' in ds_in: ds_out.SeriesTime = ds_in.SeriesTime
	if 'PerformingPhysicianName' in ds_in: ds_out.PerformingPhysicianName = ds_in.PerformingPhysicianName
	if 'ProtocolName' in ds_in: ds_out.ProtocolName = ds_in.ProtocolName
	if 'SeriesDescription' in ds_in: ds_out.SeriesDescription = ds_in.SeriesDescription
	if 'OperatorsName' in ds_in: ds_out.OperatorsName = ds_in.OperatorsName
	if 'BodyPartExamined' in ds_in: ds_out.BodyPartExamined = ds_in.BodyPartExamined
	if 'PatientPosition' in ds_in: ds_out.PatientPosition = ds_in.PatientPosition
	if 'SmallestPixelValueInSeries' in ds_in: ds_out.SmallestPixelValueInSeries = ds_in.SmallestPixelValueInSeries
	if 'LargestPixelValueInSeries' in ds_in: ds_out.LargestPixelValueInSeries = ds_in.LargestPixelValueInSeries
	# Frame of Reference Module Attributes
	ds_out.FrameOfReferenceUID       = ds_in.FrameOfReferenceUID
	if 'PositionReferenceIndicator' in ds_in: ds_out.PositionReferenceIndicator = ds_in.PositionReferenceIndicator
	# General Equipment Module Attributes
	if 'Manufacturer' in ds_in: ds_out.Manufacturer = ds_in.Manufacturer
	if 'InstitutionName' in ds_in: ds_out.InstitutionName = ds_in.InstitutionName
	if 'InstitutionAddress' in ds_in: ds_out.InstitutionAddress = ds_in.InstitutionAddress
	if 'StationName' in ds_in: ds_out.StationName = ds_in.StationName
	if 'InstitutionalDepartmentName' in ds_in: ds_out.InstitutionalDepartmentName = ds_in.InstitutionalDepartmentName
	if 'ManufacturerModelName' in ds_in: ds_out.ManufacturerModelName = ds_in.ManufacturerModelName
	if 'DeviceSerialNumber' in ds_in: ds_out.DeviceSerialNumber = ds_in.DeviceSerialNumber
	if 'SoftwareVersions' in ds_in: ds_out.SoftwareVersions = ds_in.SoftwareVersions
	if 'SpatialResolution' in ds_in: ds_out.SpatialResolution = ds_in.SpatialResolution
	if 'DateOfLastCalibration' in ds_in: ds_out.DateOfLastCalibration = ds_in.DateOfLastCalibration
	if 'TimeOfLastCalibration' in ds_in: ds_out.TimeOfLastCalibration = ds_in.TimeOfLastCalibration
	if 'PixelPaddingValue' in ds_in: ds_out.PixelPaddingValue = ds_in.PixelPaddingValue
	# General Image Module Attributes
	if 'InstanceNumber' in ds_in: ds_out.InstanceNumber = ds_in.InstanceNumber
	if 'PatientOrientation' in ds_in: ds_out.PatientOrientation = ds_in.PatientOrientation
	if 'ContentDate' in ds_in: ds_out.ContentDate = ds_in.ContentDate
	if 'ContentTime' in ds_in: ds_out.ContentTime = ds_in.ContentTime
	if 'ImageType' in ds_in: ds_out.ImageType = ds_in.ImageType
	if 'AcquisitionNumber' in ds_in: ds_out.AcquisitionNumber = ds_in.AcquisitionNumber
	if 'AcquisitionDate' in ds_in: ds_out.AcquisitionDate = ds_in.AcquisitionDate
	if 'AcquisitionTime' in ds_in: ds_out.AcquisitionTime = ds_in.AcquisitionTime
	if 'DerivationDescription' in ds_in: ds_out.DerivationDescription = ds_in.DerivationDescription
	if 'ImagesInAcquisition' in ds_in: ds_out.ImagesInAcquisition = ds_in.ImagesInAcquisition
	if 'ImageComments' in ds_in: ds_out.ImageComments = ds_in.ImageComments
	if 'QualityControlImage' in ds_in: ds_out.QualityControlImage = ds_in.QualityControlImage
	if 'BurnedInAnnotation' in ds_in: ds_out.BurnedInAnnotation = ds_in.BurnedInAnnotation
	if 'RecognizableVisualFeatures' in ds_in: ds_out.RecognizableVisualFeatures = ds_in.RecognizableVisualFeatures
	if 'LossyImageCompression' in ds_in: ds_out.LossyImageCompression = ds_in.LossyImageCompression
	if 'LossyImageCompressionRatio' in ds_in: ds_out.LossyImageCompressionRatio = ds_in.LossyImageCompressionRatio
	if 'LossyImageCompressionMethod' in ds_in: ds_out.LossyImageCompressionMethod = ds_in.LossyImageCompressionMethod
	# Image Plane Module Attributes
	ds_out.PixelSpacing              = ds_in.PixelSpacing
	ds_out.ImageOrientationPatient   = ds_in.ImageOrientationPatient
	ds_out.ImagePositionPatient      = ds_in.ImagePositionPatient
	if 'SliceThickness' in ds_in: ds_out.SliceThickness = ds_in.SliceThickness
	if 'SliceLocation' in ds_in: ds_out.SliceLocation = ds_in.SliceLocation
	# Image Pixel Module Attributes
	ds_out.SamplesPerPixel           = ds_in.SamplesPerPixel
	ds_out.PhotometricInterpretation = ds_in.PhotometricInterpretation
	ds_out.Rows                      = ds_in.Rows
	ds_out.Columns                   = ds_in.Columns
	ds_out.BitsAllocated             = ds_in.BitsAllocated
	ds_out.BitsStored                = ds_in.BitsStored
	ds_out.HighBit                   = ds_in.HighBit
	ds_out.PixelRepresentation       = ds_in.PixelRepresentation
	if 'PlanarConfiguration' in ds_in: ds_out.PlanarConfiguration = ds_in.PlanarConfiguration
	if 'PixelAspectRatio' in ds_in: ds_out.PixelAspectRatio = ds_in.PixelAspectRatio
	if 'SmallestImagePixelValue' in ds_in: ds_out.SmallestImagePixelValue = ds_in.SmallestImagePixelValue
	if 'LargestImagePixelValue' in ds_in: ds_out.LargestImagePixelValue = ds_in.LargestImagePixelValue
	if 'RedPaletteColorLookupTableDescriptor' in ds_in: ds_out.RedPaletteColorLookupTableDescriptor = ds_in.RedPaletteColorLookupTableDescriptor
	if 'GreenPaletteColorLookupTableDescriptor' in ds_in: ds_out.GreenPaletteColorLookupTableDescriptor = ds_in.GreenPaletteColorLookupTableDescriptor
	if 'BluePaletteColorLookupTableDescriptor' in ds_in: ds_out.BluePaletteColorLookupTableDescriptor = ds_in.BluePaletteColorLookupTableDescriptor
	if 'RedPaletteColorLookupTableData' in ds_in: ds_out.RedPaletteColorLookupTableData = ds_in.RedPaletteColorLookupTableData
	if 'GreenPaletteColorLookupTableData' in ds_in: ds_out.GreenPaletteColorLookupTableData = ds_in.GreenPaletteColorLookupTableData
	if 'BluePaletteColorLookupTableData' in ds_in: ds_out.BluePaletteColorLookupTableData = ds_in.BluePaletteColorLookupTableData
	# Contrast/Bolus Module Attributes
	if 'ContrastBolusAgent' in ds_in: ds_out.ContrastBolusAgent = ds_in.ContrastBolusAgent
	if 'ContrastBolusRoute' in ds_in: ds_out.ContrastBolusRoute = ds_in.ContrastBolusRoute
	if 'ContrastBolusVolume' in ds_in: ds_out.ContrastBolusVolume = ds_in.ContrastBolusVolume
	if 'ContrastBolusStartTime' in ds_in: ds_out.ContrastBolusStartTime = ds_in.ContrastBolusStartTime
	if 'ContrastBolusStopTime' in ds_in: ds_out.ContrastBolusStopTime = ds_in.ContrastBolusStopTime
	if 'ContrastBolusTotalDose' in ds_in: ds_out.ContrastBolusTotalDose = ds_in.ContrastBolusTotalDose
	if 'ContrastFlowRates' in ds_in: ds_out.ContrastFlowRates = ds_in.ContrastFlowRates
	if 'ContrastFlowDurations' in ds_in: ds_out.ContrastFlowDurations = ds_in.ContrastFlowDurations
	if 'ContrastBolusIngredient' in ds_in: ds_out.ContrastBolusIngredient = ds_in.ContrastBolusIngredient
	if 'ContrastBolusIngredientConcentration' in ds_in: ds_out.ContrastBolusIngredientConcentration = ds_in.ContrastBolusIngredientConcentration
	# Cine Module Attributes
	if 'PreferredPlaybackSequencing' in ds_in: ds_out.PreferredPlaybackSequencing = ds_in.PreferredPlaybackSequencing
	if 'FrameTime' in ds_in: ds_out.FrameTime = ds_in.FrameTime
	if 'FrameTimeVector' in ds_in: ds_out.FrameTimeVector = ds_in.FrameTimeVector
	if 'StartTrim' in ds_in: ds_out.StartTrim = ds_in.StartTrim
	if 'StopTrim' in ds_in: ds_out.StopTrim = ds_in.StopTrim
	if 'RecommendedDisplayFrameRate' in ds_in: ds_out.RecommendedDisplayFrameRate = ds_in.RecommendedDisplayFrameRate
	if 'CineRate' in ds_in: ds_out.CineRate = ds_in.CineRate
	if 'FrameDelay' in ds_in: ds_out.FrameDelay = ds_in.FrameDelay
	if 'EffectiveDuration' in ds_in: ds_out.EffectiveDuration = ds_in.EffectiveDuration
	if 'ActualFrameDuration' in ds_in: ds_out.ActualFrameDuration = ds_in.ActualFrameDuration
	# Multi-Frame Module Attributes
	if 'NumberOfFrames' in ds_in: ds_out.NumberOfFrames = ds_in.NumberOfFrames
	if 'FrameIncrementPointer' in ds_in: ds_out.FrameIncrementPointer = ds_in.FrameIncrementPointer
	# Bi-Plane Sequence Module Attributes
	if 'Planes' in ds_in: ds_out.Planes = ds_in.Planes
	if 'BiPlaneAcquisitionSequence' in ds_in: ds_out.BiPlaneAcquisitionSequence = ds_in.BiPlaneAcquisitionSequence
	# Bi-Plane Image Module Attributes
	if 'SmallestImagePixelValueInPlane' in ds_in: ds_out.SmallestImagePixelValueInPlane = ds_in.SmallestImagePixelValueInPlane
	if 'LargestImagePixelValueInPlane' in ds_in: ds_out.LargestImagePixelValueInPlane = ds_in.LargestImagePixelValueInPlane
	# Frame Pointers Module Attributes
	if 'RepresentativeFrameNumber' in ds_in: ds_out.RepresentativeFrameNumber = ds_in.RepresentativeFrameNumber
	if 'FrameNumbersOfInterest' in ds_in: ds_out.FrameNumbersOfInterest = ds_in.FrameNumbersOfInterest
	if 'FrameOfInterestDescription' in ds_in: ds_out.FrameOfInterestDescription = ds_in.FrameOfInterestDescription
	# CT Image Module Attributes
	ds_out.ImageType                 = ds_in.ImageType
	ds_out.SamplesPerPixel           = ds_in.SamplesPerPixel
	ds_out.PhotometricInterpretation = ds_in.PhotometricInterpretation
	ds_out.BitsAllocated             = ds_in.BitsAllocated
	ds_out.BitsStored                = ds_in.BitsStored
	ds_out.HighBit                   = ds_in.HighBit
	if 'RescaleIntercept' in ds_in: ds_out.RescaleIntercept = ds_in.RescaleIntercept
	if 'RescaleSlope' in ds_in: ds_out.RescaleSlope = ds_in.RescaleSlope
	if 'KVP' in ds_in: ds_out.KVP = ds_in.KVP
	if 'AcquisitionNumber' in ds_in: ds_out.AcquisitionNumber = ds_in.AcquisitionNumber
	if 'ScanOptions' in ds_in: ds_out.ScanOptions = ds_in.ScanOptions
	if 'DataCollectionDiameter' in ds_in: ds_out.DataCollectionDiameteDataCollectionDiameter = ds_in.DataCollectionDiameter
	if 'ReconstructionDiameter' in ds_in: ds_out.ReconstructionDiameter = ds_in.ReconstructionDiameter
	if 'DistanceSourceToDetector' in ds_in: ds_out.DistanceSourceToDetector = ds_in.DistanceSourceToDetector
	if 'DistanceSourceToPatient' in ds_in: ds_out.DistanceSourceToPatient = ds_in.DistanceSourceToPatient
	if 'GantryDetectorTilt' in ds_in: ds_out.GantryDetectorTilt = ds_in.GantryDetectorTilt
	if 'TableHeight' in ds_in: ds_out.TableHeight = ds_in.TableHeight
	if 'RotationDirection' in ds_in: ds_out.RotationDirection = ds_in.RotationDirection
	if 'ExposureTime' in ds_in: ds_out.ExposureTime = ds_in.ExposureTime
	if 'XrayTubeCurrent' in ds_in: ds_out.XrayTubeCurrent = ds_in.XrayTubeCurrent
	if 'Exposure' in ds_in: ds_out.Exposure = ds_in.Exposure
	if 'FilterType' in ds_in: ds_out.FilterType = ds_in.FilterType
	if 'GeneratorPower' in ds_in: ds_out.GeneratorPower = ds_in.GeneratorPower
	if 'FocalSpot' in ds_in: ds_out.FocalSpot = ds_in.FocalSpot
	if 'ConvolutionKernel' in ds_in: ds_out.ConvolutionKernel = ds_in.ConvolutionKernel
	# MR Image Module Attributes
	ds_out.ImageType                 = ds_in.ImageType
	ds_out.SamplesPerPixel           = ds_in.SamplesPerPixel
	ds_out.PhotometricInterpretation = ds_in.PhotometricInterpretation
	ds_out.BitsAllocated             = ds_in.BitsAllocated
	if 'ScanningSequence' in ds_in: ds_out.ScanningSequence = ds_in.ScanningSequence
	if 'SequenceVariant' in ds_in: ds_out.SequenceVariant = ds_in.SequenceVariant
	if 'ScanOptions' in ds_in: ds_out.ScanOptions = ds_in.ScanOptions
	if 'MRAcquisitionType' in ds_in: ds_out.MRAcquisitionType = ds_in.MRAcquisitionType
	if 'RepetitionTime' in ds_in: ds_out.RepetitionTime = ds_in.RepetitionTime
	if 'EchoTime' in ds_in: ds_out.EchoTime = ds_in.EchoTime
	if 'EchoTrainLength' in ds_in: ds_out.EchoTrainLength = ds_in.EchoTrainLength
	if 'InversionTime' in ds_in: ds_out.InversionTime = ds_in.InversionTime
	if 'TriggerTime' in ds_in: ds_out.TriggerTime = ds_in.TriggerTime
	if 'SequenceName' in ds_in: ds_out.SequenceName = ds_in.SequenceName
	if 'AngioFlag' in ds_in: ds_out.AngioFlag = ds_in.AngioFlag
	if 'NumberOfAverages' in ds_in: ds_out.NumberOfAverages = ds_in.NumberOfAverages
	if 'ImagingFrequency' in ds_in: ds_out.ImagingFrequency = ds_in.ImagingFrequency
	if 'ImagedNucleus' in ds_in: ds_out.ImagedNucleus = ds_in.ImagedNucleus
	if 'EchoNumbers' in ds_in: ds_out.EchoNumbers = ds_in.EchoNumbers
	if 'MagneticFieldStrength' in ds_in: ds_out.MagneticFieldStrength = ds_in.MagneticFieldStrength
	if 'SpacingBetweenSlices' in ds_in: ds_out.SpacingBetweenSlices = ds_in.SpacingBetweenSlices
	if 'NumberOfPhaseEncodingSteps' in ds_in: ds_out.NumberOfPhaseEncodingSteps = ds_in.NumberOfPhaseEncodingSteps
	if 'PercentSampling' in ds_in: ds_out.PercentSampling = ds_in.PercentSampling
	if 'PercentPhaseFieldOfView' in ds_in: ds_out.PercentPhaseFieldOfView = ds_in.PercentPhaseFieldOfView
	if 'PixelBandwidth' in ds_in: ds_out.PixelBandwidth = ds_in.PixelBandwidth
	if 'NominalInterval' in ds_in: ds_out.NominalInterval = ds_in.NominalInterval
	if 'BeatRejectionFlag' in ds_in: ds_out.BeatRejectionFlag = ds_in.BeatRejectionFlag
	if 'LowRRValue' in ds_in: ds_out.LowRRValue = ds_in.LowRRValue
	if 'HighRRValue' in ds_in: ds_out.HighRRValue = ds_in.HighRRValue
	if 'IntervalsAcquired' in ds_in: ds_out.IntervalsAcquired = ds_in.IntervalsAcquired
	if 'IntervalsRejected' in ds_in: ds_out.IntervalsRejected = ds_in.IntervalsRejected
	if 'PVCRejection' in ds_in: ds_out.PVCRejection = ds_in.PVCRejection
	if 'SkipBeats' in ds_in: ds_out.SkipBeats = ds_in.SkipBeats
	if 'HeartRate' in ds_in: ds_out.HeartRate = ds_in.HeartRate
	if 'CardiacNumberOfImages' in ds_in: ds_out.CardiacNumberOfImages = ds_in.CardiacNumberOfImages
	if 'TriggerWindow' in ds_in: ds_out.TriggerWindow = ds_in.TriggerWindow
	if 'ReconstructionDiameter' in ds_in: ds_out.ReconstructionDiameter = ds_in.ReconstructionDiameter
	if 'ReceivingCoil' in ds_in: ds_out.ReceivingCoil = ds_in.ReceivingCoil
	if 'TransmittingCoil' in ds_in: ds_out.TransmittingCoil = ds_in.TransmittingCoil
	if 'AcquisitionMatrix' in ds_in: ds_out.AcquisitionMatrix = ds_in.AcquisitionMatrix
	if 'PhaseEncodingDirection' in ds_in: ds_out.PhaseEncodingDirection = ds_in.PhaseEncodingDirection
	if 'FlipAngle' in ds_in: ds_out.FlipAngle = ds_in.FlipAngle
	if 'SAR' in ds_in: ds_out.SAR = ds_in.SAR
	if 'VariableFlipAngleFlag' in ds_in: ds_out.VariableFlipAngleFlag = ds_in.VariableFlipAngleFlag
	if 'dBdt' in ds_in: ds_out.dBdt = ds_in.dBdt
	if 'TemporalPositionIdentifier' in ds_in: ds_out.TemporalPositionIdentifier = ds_in.TemporalPositionIdentifier
	if 'NumberOfTemporalPositions' in ds_in: ds_out.NumberOfTemporalPositions = ds_in.NumberOfTemporalPositions
	if 'TemporalResolution' in ds_in: ds_out.TemporalResolution = ds_in.TemporalResolution
	if 'ImageLaterality' in ds_in: ds_out.ImageLaterality = ds_in.ImageLaterality

if __name__ == '__main__':
	exit(0)
