def scores_to_file(scores, filename):
    """Saves scores list to file"""
    with open(filename, 'w') as f:
        for s in scores:
            f.write("%s\n" % s)
