"""Tests for plotting functions."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
from real_simple_stats.plots import (
    set_minimalist_style,
    plot_norm_hist,
    plot_box,
    plot_observed_vs_expected,
)


class TestSetMinimalistStyle:
    @patch("matplotlib.pyplot.rcParams")
    def test_set_minimalist_style(self, mock_rcparams):
        set_minimalist_style()
        # Verify that rcParams.update was called
        assert mock_rcparams.update.called


class TestPlotNormHist:
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.axvline")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.hist")
    @patch("matplotlib.pyplot.title")
    def test_plot_norm_hist_all_features(
        self, mock_title, mock_hist, mock_plot, mock_axvline, mock_savefig, mock_show
    ):
        data = np.random.normal(50, 10, 100)
        mean = 50
        std = 10

        # Mock hist to return bins_edges
        mock_hist.return_value = (None, np.linspace(20, 80, 31), None)

        plot_norm_hist(data, mean, std, bins=30, show_pdf=True, show_lines=True, title=True)

        # Verify hist was called
        assert mock_hist.called
        # Verify PDF was plotted
        assert mock_plot.called
        # Verify vertical lines were added (2 calls for ±2σ)
        assert mock_axvline.call_count == 2
        # Verify title was set
        assert mock_title.called
        # Verify savefig and show were called
        assert mock_savefig.called
        assert mock_show.called

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.axvline")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.hist")
    @patch("matplotlib.pyplot.title")
    def test_plot_norm_hist_no_pdf(
        self, mock_title, mock_hist, mock_plot, mock_axvline, mock_savefig, mock_show
    ):
        data = np.random.normal(50, 10, 100)
        mean = 50
        std = 10

        mock_hist.return_value = (None, np.linspace(20, 80, 31), None)

        plot_norm_hist(data, mean, std, show_pdf=False, show_lines=True, title=True)

        # Verify PDF was NOT plotted
        assert not mock_plot.called
        # Verify lines were still added
        assert mock_axvline.call_count == 2

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.axvline")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.hist")
    @patch("matplotlib.pyplot.title")
    def test_plot_norm_hist_no_lines(
        self, mock_title, mock_hist, mock_plot, mock_axvline, mock_savefig, mock_show
    ):
        data = np.random.normal(50, 10, 100)
        mean = 50
        std = 10

        mock_hist.return_value = (None, np.linspace(20, 80, 31), None)

        plot_norm_hist(data, mean, std, show_pdf=True, show_lines=False, title=True)

        # Verify lines were NOT added
        assert not mock_axvline.called
        # Verify PDF was still plotted
        assert mock_plot.called

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.axvline")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.hist")
    @patch("matplotlib.pyplot.title")
    def test_plot_norm_hist_no_title(
        self, mock_title, mock_hist, mock_plot, mock_axvline, mock_savefig, mock_show
    ):
        data = np.random.normal(50, 10, 100)
        mean = 50
        std = 10

        mock_hist.return_value = (None, np.linspace(20, 80, 31), None)

        plot_norm_hist(data, mean, std, show_pdf=True, show_lines=True, title=False)

        # Verify title was NOT set
        assert not mock_title.called

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.hist")
    def test_plot_norm_hist_custom_bins(self, mock_hist, mock_savefig, mock_show):
        data = np.random.normal(50, 10, 100)
        mean = 50
        std = 10

        mock_hist.return_value = (None, np.linspace(20, 80, 51), None)

        plot_norm_hist(data, mean, std, bins=50, show_pdf=False, show_lines=False, title=False)

        # Verify bins parameter was passed
        call_args = mock_hist.call_args
        assert call_args[1]["bins"] == 50


class TestPlotBox:
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    def test_plot_box_default(self, mock_subplots, mock_savefig, mock_show):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Mock the axes object
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plot_box(data)

        # Verify boxplot was called
        assert mock_ax.boxplot.called
        # Verify it was called with vert=False (horizontal)
        call_args = mock_ax.boxplot.call_args
        assert call_args[1]["vert"] is False
        # Verify showfliers=False by default
        assert call_args[1]["showfliers"] is False
        # Verify savefig and show were called
        assert mock_savefig.called
        assert mock_show.called

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    def test_plot_box_with_outliers(self, mock_subplots, mock_savefig, mock_show):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]

        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plot_box(data, showfliers=True)

        # Verify showfliers=True was passed
        call_args = mock_ax.boxplot.call_args
        assert call_args[1]["showfliers"] is True

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    def test_plot_box_styling(self, mock_subplots, mock_savefig, mock_show):
        data = [1, 2, 3, 4, 5]

        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plot_box(data)

        # Verify styling parameters were passed
        call_args = mock_ax.boxplot.call_args
        assert call_args[1]["patch_artist"] is True
        assert "boxprops" in call_args[1]
        assert "whiskerprops" in call_args[1]


class TestPlotObservedVsExpected:
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.legend")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.bar")
    def test_plot_observed_vs_expected_default(
        self, mock_bar, mock_xticks, mock_title, mock_legend, mock_savefig, mock_show
    ):
        observed = [10, 20, 30, 40]
        expected = [12, 18, 32, 38]

        plot_observed_vs_expected(observed, expected)

        # Verify bar was called twice (once for observed, once for expected)
        assert mock_bar.call_count == 2
        # Verify xticks was called
        assert mock_xticks.called
        # Verify title was set
        assert mock_title.called
        # Verify legend was added
        assert mock_legend.called
        # Verify savefig and show were called
        assert mock_savefig.called
        assert mock_show.called

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.legend")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.bar")
    def test_plot_observed_vs_expected_custom_title(
        self, mock_bar, mock_xticks, mock_title, mock_legend, mock_savefig, mock_show
    ):
        observed = [10, 20, 30]
        expected = [12, 18, 32]

        plot_observed_vs_expected(observed, expected, title="Custom Title")

        # Verify custom title was used
        mock_title.assert_called_once_with("Custom Title")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.legend")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.bar")
    def test_plot_observed_vs_expected_different_lengths(
        self, mock_bar, mock_xticks, mock_title, mock_legend, mock_savefig, mock_show
    ):
        observed = [10, 20]
        expected = [12, 18]

        plot_observed_vs_expected(observed, expected)

        # Verify it handles different data lengths
        assert mock_bar.call_count == 2
        # Verify xticks uses correct range
        call_args = mock_xticks.call_args
        assert len(call_args[0][0]) == 2

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.bar")
    def test_plot_observed_vs_expected_bar_properties(
        self, mock_bar, mock_savefig, mock_show
    ):
        observed = [10, 20, 30]
        expected = [12, 18, 32]

        plot_observed_vs_expected(observed, expected)

        # Verify bar styling
        calls = mock_bar.call_args_list
        # First call (observed) should have label="Observed"
        assert calls[0][1]["label"] == "Observed"
        # Second call (expected) should have label="Expected"
        assert calls[1][1]["label"] == "Expected"
