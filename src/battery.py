class Battery:
    """A simple battery model for energy storage for a microgrid. It can charge and discharge energy, 
    and has a state of charge (SOC) that is updated based on the energy flow."""


    def __init__(self, name, capacity_mwh=100, max_rate_mw=25, efficiency=0.92, available=True):
        self.name = name            # name of the battery
        self.capacity = capacity_mwh   # how much energy it can hold
        self.max_rate = max_rate_mw     # max charge/discharge speed
        self.eff = efficiency          
        self.soc = 0.5 * capacity_mwh     # current SOC (start at half of capacity) so that we can test both direction i.e. half full or half empty, we can take it 0 as well.
        self.available = available    # True means working, False means broken

    def charge(self, mw):
        """Try to charge at 'mw' for 15min. Return MW actually used."""
        if not self.available:   # check if the battery is not working or not available
            return 0
        mw = min(mw, self.max_rate)          # limit charge speed
        room = self.capacity - self.soc                 # check remaining space in battery, measured in MWh
        mw = min(mw, room / (0.25 * self.eff))            # prevent overfilling
        self.soc += mw * 0.25 * self.eff                # update battery level(increase SOC)
        return mw

    def discharge(self, mw):
        """Try to discharge 'mw' for 15min. Return MW actually delivered"""
        if not self.available:
            return 0

        mw = min(mw, self.max_rate)      # limit discharge speed
        mw = min(mw, self.soc / 0.25)        # check how much energy is stored
        self.soc -= mw * 0.25           # decrease SOC
        return mw

if __name__ == "__main__":
    b3 = Battery(
        name="B3",
        capacity_mwh=150,
        max_rate_mw=40,
        efficiency=0.95,
        available=True
    )
    print(b3.charge(80))
    print(b3.discharge(20))