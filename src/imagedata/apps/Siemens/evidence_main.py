
if __name__ == '__main__':
	global separate_laterality

	image_data.scan_options()
	scan_local_options()
	try:
		out_name = sys.argv[1]
		evidence_name = sys.argv[2]
		in_dirs = sys.argv[3:]
	except Exception:
		print("Usage:",sys.argv[0], "<options> [-laterality sep] out evidence in...")
		print("\nWhere <options> are:\n", image_data.options_to_text())
		print("  [-laterality separate|combined] - Separate left and right ROIs in separate files. Other combine both (default).")
		sys.exit(1)

	try:
		hdr, si = imagedata.readdata.read(in_dirs, imagedata.formats.INPUT_ORDER_TIME)
	except UnknownInputError:
		print("Could not determine input format of %s." % in_dirs[1])
		import traceback
		traceback.print_exc(file=sys.stdout)
		sys.exit(1)

	try:
		reading = imagedata.read_headers((evidence_name,), imagedata.formats.INPUT_ORDER_NONE, force_order=True)
	except image_data.UnknownInputError:
		print("Could not determine input format of %s." % evidence_name)
		import traceback
		traceback.print_exc(file=sys.stdout)
		sys.exit(1)

	mask = {}
	mask,content = evidence2mask(mask, reading, hdr, si, separate_laterality)
	print(content)

	serNum = 5000
	imageType = ['DERIVED', 'SECONDARY', 'MASK']

	f = open(os.path.join(out_name, "timepoint.txt"), "w")
	f.write("%d" % content['timepoint'])
	f.close()

	for laterality in mask.keys():
		hdr_copy = hdr.copy()
		hdr_copy.removePrivateTags()
		hdr_copy.setSeriesNumber(serNum)
		hdr_copy.setSeriesDescription("Mask %s" % laterality)
		hdr_copy.setImageType(imageType)
		# Acquisition Date
		hdr_copy.setDicomAttribute((0x0008,0x0022), content['date'])
		# Acquisition Time
		hdr_copy.setDicomAttribute((0x0008,0x0032), content['time'])
		# Operator's Name
		hdr_copy.setDicomAttribute((0x0008,0x1070), content['creator'])
		# Image Laterality
		hdr_copy.setDicomAttribute((0x0020,0x0062), laterality)
		image_data.write_images(hdr_copy, mask[laterality], os.path.join(out_name, "%s_%%05d" % laterality))
		serNum += 1

