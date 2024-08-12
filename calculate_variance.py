####
import netket as nk
def calculate_variance(ha,vs):
    return vs.expect(ha@ha).mean.real-vs.expect(ha).mean.real**2
