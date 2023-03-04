"""
# * Code Reference: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html
"""

# %%
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from soupsieve import select


class IrrigationControlSystem:
    def __init__(self):
        # * Task 1.1 Generate fuzzy membership functions
        self.define_variable_range()
        self.generate_membership_func()
        self.view_membership_func()

    def run(self, soil_moisture: float, air_humidity: float, temperature: float):
        # * Task 1.2 Fuzzification and fuzzy rules evaluation

        self.fuzzification(soil_moisture, air_humidity, temperature)
        self.rules_evaluation()
        self.view_output_membership()

        # * Task 1.3 Rule Aggregation
        self.rule_aggregation()

        # * Task 1.4 Defuzzification
        self.defuzzification()
        self.defuzzification_plot()

        return self.duration

    def define_variable_range(self):
        """
        Universal Variables
        #   * Soil moisture is measured in percentage and ranges from 0 to 100%
        #   * Relative humidity is measured in percentage and ranges from 0 to 100%
        #   * Temperature has a range of [-10, 50] in units of degree celcius
        #   * Duration of irrigation is measured in minutes and ranges from 0 to 10
        """
        self.x_moisture = np.arange(0, 101, 1)
        self.x_humidity = np.arange(0, 101, 1)
        self.x_temperature = np.arange(-10, 51, 1)
        self.x_duration = np.arange(0, 10.5, 0.5)

    def generate_membership_func(self):
        """
        Fuzzy Membership Functions
        #   * State Variables
        #   Soil Moisture: Dry (d), Medium (m), Wet (w)
        #   Air Humidity: Dry (d), Humid (h), Wet (w)
        #   Temperature: Very Low (vl), Low (l), Normal (n), High (h), Very High (vh)

        #   * Control Variables
        #   Duration: Very Short (vs), Short (s), Medium (m), Long (l), Very Long (vl)
        """

        self.moisture_l = fuzz.trapmf(self.x_moisture, [0, 0, 20, 40])
        self.moisture_m = fuzz.trimf(self.x_moisture, [30, 50, 70])
        self.moisture_h = fuzz.trapmf(self.x_moisture, [60, 80, 100, 100])

        self.humidity_l = fuzz.trapmf(self.x_humidity, [0, 0, 15, 40])
        self.humidity_m = fuzz.trimf(self.x_humidity, [25, 50, 75])
        self.humidity_h = fuzz.trapmf(self.x_humidity, [60, 85, 100, 100])

        self.temp_l = fuzz.trapmf(self.x_temperature, [-10, -10, 0, 15])
        self.temp_m = fuzz.trimf(self.x_temperature, [10, 20, 30])
        self.temp_h = fuzz.trapmf(self.x_temperature, [25, 40, 50, 50])

        self.duration_vs = fuzz.trimf(self.x_duration, [0, 0, 2.5])
        self.duration_s = fuzz.trimf(self.x_duration, [0, 2.5, 5])
        self.duration_m = fuzz.trimf(self.x_duration, [2.5, 5, 7.5])
        self.duration_l = fuzz.trimf(self.x_duration, [5, 7.5, 10])
        self.duration_vl = fuzz.trimf(self.x_duration, [7.5, 10, 10])

    def view_membership_func(self):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

        ax0.plot(self.x_moisture, self.moisture_l,
                 'c', linewidth=1.5, label='Low')
        ax0.plot(self.x_moisture, self.moisture_m,
                 'y', linewidth=1.5, label='Medium')
        ax0.plot(self.x_moisture, self.moisture_h,
                 'm', linewidth=1.5, label='High')
        ax0.set_title('Soil Moisture')
        ax0.legend()

        ax1.plot(self.x_humidity, self.humidity_l,
                 'c', linewidth=1.5, label='Low')
        ax1.plot(self.x_humidity, self.humidity_m,
                 'y', linewidth=1.5, label='Medium')
        ax1.plot(self.x_humidity, self.humidity_h,
                 'm', linewidth=1.5, label='High')
        ax1.set_title('Air Humidity')
        ax1.legend()

        ax2.plot(self.x_temperature, self.temp_l,
                 'c', linewidth=1.5, label='Low')
        ax2.plot(self.x_temperature, self.temp_m,
                 'y', linewidth=1.5, label='Medium')
        ax2.plot(self.x_temperature, self.temp_h,
                 'm', linewidth=1.5, label='High')
        ax2.set_title('Temperature')
        ax2.legend()

        ax3.plot(self.x_duration, self.duration_vs,
                 'b', linewidth=1.5, label='Very Short')
        ax3.plot(self.x_duration,  self.duration_s,
                 'c', linewidth=1.5, label='Short')
        ax3.plot(self.x_duration,  self.duration_m,
                 'y', linewidth=1.5, label='Medium')
        ax3.plot(self.x_duration,  self.duration_l,
                 'm', linewidth=1.5, label='Long')
        ax3.plot(self.x_duration,  self.duration_vl,
                 'r', linewidth=1.5, label='Very Long')
        ax3.set_title('Duration')
        ax3.legend()

        # Turn off top/right axes
        for ax in (ax0, ax1, ax2, ax3):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()

    def fuzzification(self, soil_moisture, humidity, temperature):
        self.moisture_level_l = fuzz.interp_membership(
            self.x_moisture, self.moisture_l, soil_moisture)
        self.moisture_level_m = fuzz.interp_membership(
            self.x_moisture, self.moisture_m, soil_moisture)
        self.moisture_level_h = fuzz.interp_membership(
            self.x_moisture, self.moisture_h, soil_moisture)

        self.humidity_level_l = fuzz.interp_membership(
            self.x_humidity, self.humidity_l, humidity)
        self.humidity_level_m = fuzz.interp_membership(
            self.x_humidity, self.humidity_m, humidity)
        self.humidity_level_h = fuzz.interp_membership(
            self.x_humidity, self.humidity_h, humidity)

        self.temp_level_l = fuzz.interp_membership(
            self.x_temperature, self.temp_l, temperature)
        self.temp_level_m = fuzz.interp_membership(
            self.x_temperature, self.temp_m, temperature)
        self.temp_level_h = fuzz.interp_membership(
            self.x_temperature, self.temp_h, temperature)

    def rules_evaluation(self):
        # Rule 1:
        # (humidity_h & moisture_h & temp_h)
        # (humidity_h & moisture_h & temp_l)
        # (humidity_h & moisture_h & temp_m)
        # -> duration_vs
        active_rule1 = np.fmin(self.humidity_level_h, self.moisture_level_h)
        self.duration_activation_vs = np.fmin(active_rule1, self.duration_vs)

        # Rule 2:
        # (humidity_h & moisture_m & temp_m)
        # (humidity_m & moisture_h & temp_m)
        # (humidity_h & moisture_m & temp_l)
        # (humidity_m & moisture_h & temp_l)
        # (humidity_l & moisture_h & temp_l)
        # (humidity_h & moisture_l & temp_l)
        # -> duration_s
        condition_1 = np.fmin(
            np.fmin(self.humidity_level_h, self.moisture_level_m), self.temp_level_m)
        condition_2 = np.fmin(
            np.fmin(self.humidity_level_m, self.moisture_level_h), self.temp_level_m)
        condition_3 = np.fmin(
            np.fmin(self.humidity_level_h, self.moisture_level_m), self.temp_level_l)
        condition_4 = np.fmin(
            np.fmin(self.humidity_level_m, self.moisture_level_h), self.temp_level_l)
        condition_5 = np.fmin(
            np.fmin(self.humidity_level_l, self.moisture_level_h), self.temp_level_l)
        condition_6 = np.fmin(
            np.fmin(self.humidity_level_h, self.moisture_level_h), self.temp_level_l)
        active_rule2 = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(
            condition_1, condition_2), condition_3), condition_4), condition_5), condition_6)
        self.duration_activation_s = np.fmin(active_rule2, self.duration_s)

        # Rule 3:
        # (humidity_m & moisture_m & temp_m)
        # (humidity_m & moisture_m & temp_l)
        # (humidity_m & moisture_m & temp_h)
        # (humidity_h & moisture_m & temp_h)
        # (humidity_m & moisture_h & temp_h)
        # (humidity_l & moisture_m & temp_l)
        # (humidity_m & moisture_l & temp_l)
        # (humidity_l & moisture_h & temp_m)
        # (humidity_h & moisture_l & temp_m)
        # -> duration_m
        condition_1 = np.fmin(
            np.fmin(self.humidity_level_m, self.moisture_level_m), self.temp_level_m)
        condition_2 = np.fmin(
            np.fmin(self.humidity_level_m, self.moisture_level_m), self.temp_level_l)
        condition_3 = np.fmin(
            np.fmin(self.humidity_level_m, self.moisture_level_m), self.temp_level_h)
        condition_4 = np.fmin(
            np.fmin(self.humidity_level_h, self.moisture_level_m), self.temp_level_h)
        condition_5 = np.fmin(
            np.fmin(self.humidity_level_m, self.moisture_level_h), self.temp_level_h)
        condition_6 = np.fmin(
            np.fmin(self.humidity_level_l, self.moisture_level_m), self.temp_level_l)
        condition_7 = np.fmin(
            np.fmin(self.humidity_level_m, self.moisture_level_l), self.temp_level_l)
        condition_8 = np.fmin(
            np.fmin(self.humidity_level_l, self.moisture_level_h), self.temp_level_m)
        condition_9 = np.fmin(
            np.fmin(self.humidity_level_h, self.moisture_level_l), self.temp_level_m)
        active_rule3 = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(
            condition_1, condition_2), condition_3), condition_4), condition_5), condition_6), condition_7), condition_8), condition_9)
        self.duration_activation_m = np.fmin(active_rule3, self.duration_m)

        # Rule 4:
        # (humidity_m & moisture_l & temp_m)
        # (humidity_l & moisture_m & temp_m)
        # (humidity_l & moisture_m & temp_h)
        # (humidity_m & moisture_l & temp_h)
        # (humidity_h & moisture_l & temp_h)
        # (humidity_l & moisture_h & temp_h)
        # -> duration_l
        condition_1 = np.fmin(
            np.fmin(self.humidity_level_m, self.moisture_level_l), self.temp_level_m)
        condition_2 = np.fmin(
            np.fmin(self.humidity_level_l, self.moisture_level_m), self.temp_level_m)
        condition_3 = np.fmin(
            np.fmin(self.humidity_level_l, self.moisture_level_m), self.temp_level_h)
        condition_4 = np.fmin(
            np.fmin(self.humidity_level_m, self.moisture_level_l), self.temp_level_h)
        condition_5 = np.fmin(
            np.fmin(self.humidity_level_h, self.moisture_level_l), self.temp_level_h)
        condition_6 = np.fmin(
            np.fmin(self.humidity_level_l, self.moisture_level_h), self.temp_level_h)
        active_rule4 = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(
            condition_1, condition_2), condition_3), condition_4), condition_5), condition_6)
        self.duration_activation_l = np.fmin(active_rule4, self.duration_l)

        # Rule 5:
        # (humidity_l & moisture_l & temp_l)
        # (humidity_l & moisture_l & temp_m)
        # (humidity_l & moisture_l & temp_h)
        # -> duration_vl
        active_rule5 = np.fmin(self.humidity_level_l, self.moisture_level_l)
        self.duration_activation_vl = np.fmin(active_rule5, self.duration_vl)
        self.duration0 = np.zeros_like(self.x_duration)

    def view_output_membership(self):
        fig, ax0 = plt.subplots(figsize=(8, 3))

        ax0.fill_between(self.x_duration, self.duration0,
                         self.duration_activation_vs, facecolor='b', alpha=0.7)
        ax0.plot(self.x_duration, self.duration_vs,
                 'b', linewidth=0.5, linestyle='--', )

        ax0.fill_between(self.x_duration, self.duration0,
                         self.duration_activation_s, facecolor='c', alpha=0.7)
        ax0.plot(self.x_duration, self.duration_s,
                 'c', linewidth=0.5, linestyle='--')

        ax0.fill_between(self.x_duration, self.duration0,
                         self.duration_activation_m, facecolor='y', alpha=0.7)
        ax0.plot(self.x_duration, self.duration_m,
                 'y', linewidth=0.5, linestyle='--')

        ax0.fill_between(self.x_duration, self.duration0,
                         self.duration_activation_l, facecolor='m', alpha=0.7)
        ax0.plot(self.x_duration, self.duration_l,
                 'm', linewidth=0.5, linestyle='--')

        ax0.fill_between(self.x_duration, self.duration0,
                         self.duration_activation_vl, facecolor='r', alpha=0.7)
        ax0.plot(self.x_duration, self.duration_vl,
                 'r', linewidth=0.5, linestyle='--')

        ax0.set_title('Output membership activity')

        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()

    def rule_aggregation(self):
        # Aggregate all five output membership functions together

        self.aggregated = np.fmax(self.duration_activation_vs,
                                  np.fmax(self.duration_activation_s,
                                          np.fmax(self.duration_activation_m,
                                                  np.fmax(self.duration_activation_l,
                                                          self.duration_activation_vl
                                                          )
                                                  )
                                          )
                                  )

    def defuzzification(self):
        # Calculate defuzzified result
        self.duration = fuzz.defuzz(
            self.x_duration, self.aggregated, 'centroid')
        self.duration_activation = fuzz.interp_membership(
            self.x_duration, self.aggregated, self.duration)  # for plot

    def defuzzification_plot(self):
        fig, ax0 = plt.subplots(figsize=(8, 3), )

        ax0.plot(self.x_duration, self.duration_vs,
                 'b', linewidth=0.5, linestyle='--', )
        ax0.plot(self.x_duration, self.duration_s,
                 'c', linewidth=0.5, linestyle='--', )
        ax0.plot(self.x_duration, self.duration_m,
                 'y', linewidth=0.5, linestyle='--', )
        ax0.plot(self.x_duration, self.duration_l,
                 'm', linewidth=0.5, linestyle='--', )
        ax0.plot(self.x_duration, self.duration_vl,
                 'r', linewidth=0.5, linestyle='--', )
        ax0.fill_between(self.x_duration, self.duration0,
                         self.aggregated, facecolor='Orange', alpha=0.7)
        ax0.plot([self.duration, self.duration], [
                 0, self.duration_activation], 'k', linewidth=1.5, alpha=0.9)
        ax0.set_title('Aggregated membership and result (line)')

        # Turn off top/right axes
        for ax in (ax0, ):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()


system = IrrigationControlSystem()
output = system.run(50, 24, 23)


print("The duration for irrigation is " + str(round(output, 2)) + " minutes")

# %%
