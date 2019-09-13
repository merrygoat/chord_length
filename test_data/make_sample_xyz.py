# Create an XYZ file with particles on a lattice as sample data.
with open("test_cube_8.xyz", 'w') as out:
    for x in range(10):
        for y in range(10):
            for z in range(10):
                out.write("A\t{}\t{}\t{}\n".format(x, y, z))
