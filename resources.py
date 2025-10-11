_hatch(self, hatch):
        """Set the hatch style (for fills)."""
        self._hatch = hatch

    def get_hatch(self):
        """Get the current hatch style."""
        return self._hatch

    def get_hatch_path(self, density=6.0):
        """Return a `.Path` for the current hatch."""
        hatch = self.get_hatch()
        if hatch is None:
            return None
        return Path.hatch(hatch, density)

    def get_hatch_color(self):
        """Get the hatch color."""
        return self._hatch_color

    def set_hatch_color(self, hatch_color):
        """Set the hatch color."""
        self._hatch_color = hatch_color

    def get_hatch_linewidth(self):
        """Get the hatch linewidth."""
        return self._hatch_linewidth

    def set_hatch_linewidth(self, hatch_linewidth):
        """Set the hatch linewidth."""
        self._hatch_linewidth = hatch_linewidth

    def get_sketch_params(self):
        """
        Return the sketch parameters for the artist.

        Returns
        -------
        tuple or `None`

            A 3-tuple with the following elements:

            * ``scale``: The amplitude of the wiggle perpendicular to the
              source line.
            * ``length``: The length of the wiggle along the line.
            * ``randomness``: The scale factor by which the length is
              shrunken or expanded.

            May return `None` if no sketch parameters were set.
        """
        return self._sketch

    def set_sketch_params(self, scale=None, length=None, randomness=None):
        """
        Set the sketch parameters.

        Parameters
        ----------
        scale : float, optional
            The amplitude of the wiggle perpendicular to the source line, in
            pixels.  If scale is `None`, or not provided, no sketch filter will
            be provided.
        length : float, default: 128
            The length of the wiggle along the line, in pixels.
        randomness : float, default: 16
            The scale factor by which the length is shrunken or expanded.
        """
        self._sketch = (
            None if scale is None
            else (scale, length or 128., randomness or 16.))


class TimerBase:
    """
    A base class for providing timer events, useful for things animations.
    Backends need to implement a few specific methods in order to use their
    own timing mechanisms so that the timer events are integrated into their
    event loops.

    Subclasses must override the following methods:

    - ``_timer_start``: Backend-specific code for starting the timer.
    - ``_timer_stop``: Backend-specific code for stopping the timer.

    Subclasses may additionally override the following methods:

    - ``_timer_set_single_shot``: Code for setting the timer to single shot
      operating mode, if supported by the timer object.  If not, the `Timer`
      class itself will store the flag and the ``_on_timer`` method should be
      overridden to support such behavior.

    - ``_timer_set_interval``: Code for setting the interval on the timer, if
      there is a method for doing so on the timer object.

    - ``_on_timer``: The internal function that any timer object should call,
      which will handle the task of running all callbacks that have been set.
    """

    def __init__(self, interval=None, callbacks=None):
        """
        Parameters
        ----------
        interval : int, default: 1000ms
            The time between timer events in milliseconds.  Will be stored as
            ``timer.interval``.
        callbacks : list[tuple[callable, tuple, dict]]
            List of (func, args, kwargs) tuples that will be called upon timer
            events.  This list is accessible as ``timer.callbacks`` and can be
            manipulated directly, or the functions `~.TimerBase.add_callback`
            and `~.TimerBase.remove_callback` can be used.
        """
        self.callbacks = [] if callbacks is None else callbacks.copy()
        # Set .interval and not ._interval to go through the property setter.
        self.interval = 1000 if interval is None else interval
        self.single_shot = False

    def __del__(self):
        """Need to stop timer and possibly disconnect timer."""
        self._timer_stop()

    @_api.delete_parameter("3.9", "interval", alternative="timer.interval")
    def start(self, interval=None):
        """
        Start the timer object.

        Parameters
        ----------
        interval : int, optional
            Timer interval in milliseconds; overrides a previously set interval
            if provided.
        """
        if interval is not None:
            self.interval = interval
        self._timer_start()

    def stop(self):
        """Stop the timer."""
        self._timer_stop()

    def _timer_start(self):
        pass

    def _timer_stop(self):
        pass

    @property
    def interval(self):
        """The time between timer events, in milliseconds."""
        return self._interval

    @interval.setter
    def interval(self, interval):
        # Force to int since none of the backends actually support fractional
        # milliseconds, and some error or give warnings.
        # Some backends also fail when interval == 0, so ensure >= 1 msec
        interval = max(int(interval), 1)
        self._interval = interval
        self._timer_set_interval()

    @property
    def single_shot(self):
        """Whether this timer should stop after a single run."""
        return self._single

    @single_shot.setter
    def single_shot(self, ss):
        self._single = ss
        self._timer_set_single_shot()

    def add_callback(self, func, *args, **kwargs):
        """
        Register *func* to be called by timer when the event fires. Any
        additional arguments provided will be passed to *func*.

        This function returns *func*, which makes it possible to use it as a
        decorator.
        """
        self.callbacks.append((func, args, kwargs))
        return func

    def remove_callback(self, func, *args, **kwargs):
        """
        Remov