import React, { useState, useEffect } from 'react';
import { MapPin, Cloud, Wind, Compass, Clock, Droplets } from 'lucide-react';

const RainRadarCard = () => {
  const [rainData, setRainData] = useState({
    timeToRain: '--',
    distance: '--',
    speed: '--',
    direction: '--',
    bearing: '--',
    status: 'monitoring'
  });

  // Mock data simulation - replace with actual Home Assistant entity data
  useEffect(() => {
    // Simulating data updates every 3 seconds
    const interval = setInterval(() => {
      // In real implementation, fetch from Home Assistant API
      // For demo, we'll use mock data
      const mockTime = Math.floor(Math.random() * 120);
      setRainData({
        timeToRain: mockTime < 100 ? mockTime : 999,
        distance: mockTime < 100 ? (mockTime * 0.5).toFixed(1) : '--',
        speed: mockTime < 100 ? (35 + Math.random() * 15).toFixed(1) : '--',
        direction: mockTime < 100 ? (90 + Math.random() * 20).toFixed(0) : '--',
        bearing: mockTime < 100 ? (95 + Math.random() * 10).toFixed(0) : '--',
        status: mockTime === 0 ? 'raining' : mockTime < 100 ? 'approaching' : 'clear'
      });
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    switch(rainData.status) {
      case 'raining': return 'bg-blue-500';
      case 'approaching': return 'bg-yellow-500';
      default: return 'bg-green-500';
    }
  };

  const getStatusText = () => {
    if (rainData.timeToRain === 0) return 'RAINING NOW!';
    if (rainData.timeToRain === 999) return 'No rain detected';
    if (rainData.timeToRain < 30) return 'Rain approaching soon';
    return 'Rain detected';
  };

  const formatTime = (minutes) => {
    if (minutes === 0) return 'NOW';
    if (minutes === 999 || minutes === '--') return '--';
    if (minutes < 60) return `${minutes} min`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  const getDirectionName = (degrees) => {
    if (degrees === '--' || degrees === -1) return '--';
    const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                       'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
    const index = Math.round(degrees / 22.5) % 16;
    return directions[index];
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4 bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Cloud className="w-8 h-8 text-blue-400" />
          <h1 className="text-2xl font-bold text-white">Rain Predictor</h1>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${getStatusColor()} animate-pulse`}></div>
          <span className="text-sm text-gray-300">Live</span>
        </div>
      </div>

      {/* Status Banner */}
      <div className={`mb-6 p-4 rounded-xl ${
        rainData.timeToRain === 0 ? 'bg-blue-600' :
        rainData.timeToRain < 30 ? 'bg-yellow-600' :
        rainData.timeToRain === 999 ? 'bg-green-600' : 'bg-orange-600'
      } transition-colors duration-500`}>
        <div className="flex items-center justify-between">
          <span className="text-white font-semibold text-lg">{getStatusText()}</span>
          {rainData.timeToRain !== 999 && rainData.timeToRain !== '--' && (
            <span className="text-white text-2xl font-bold">
              {formatTime(rainData.timeToRain)}
            </span>
          )}
        </div>
      </div>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        {/* Time to Rain */}
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 hover:border-blue-500 transition-colors">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-5 h-5 text-blue-400" />
            <span className="text-xs text-gray-400 uppercase">Time</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {formatTime(rainData.timeToRain)}
          </div>
        </div>

        {/* Distance */}
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 hover:border-blue-500 transition-colors">
          <div className="flex items-center gap-2 mb-2">
            <MapPin className="w-5 h-5 text-green-400" />
            <span className="text-xs text-gray-400 uppercase">Distance</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {rainData.distance === '--' ? '--' : `${rainData.distance} km`}
          </div>
        </div>

        {/* Speed */}
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 hover:border-blue-500 transition-colors">
          <div className="flex items-center gap-2 mb-2">
            <Wind className="w-5 h-5 text-purple-400" />
            <span className="text-xs text-gray-400 uppercase">Speed</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {rainData.speed === '--' ? '--' : `${rainData.speed} km/h`}
          </div>
        </div>

        {/* Direction */}
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 hover:border-blue-500 transition-colors">
          <div className="flex items-center gap-2 mb-2">
            <Compass className="w-5 h-5 text-orange-400" />
            <span className="text-xs text-gray-400 uppercase">Direction</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {getDirectionName(rainData.direction)}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {rainData.direction === '--' ? '' : `${rainData.direction}°`}
          </div>
        </div>

        {/* Bearing */}
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 hover:border-blue-500 transition-colors">
          <div className="flex items-center gap-2 mb-2">
            <Compass className="w-5 h-5 text-cyan-400" />
            <span className="text-xs text-gray-400 uppercase">Bearing</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {getDirectionName(rainData.bearing)}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {rainData.bearing === '--' ? '' : `${rainData.bearing}°`}
          </div>
        </div>
      </div>

      {/* Visual Indicator */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <div className="flex items-center justify-center gap-4">
          <div className="flex flex-col items-center">
            <MapPin className="w-8 h-8 text-red-500 mb-2" />
            <span className="text-sm text-gray-400">Your Location</span>
          </div>
          
          {rainData.timeToRain !== 999 && rainData.timeToRain !== '--' && (
            <>
              <div className="flex-1 relative h-2 bg-slate-700 rounded-full overflow-hidden">
                <div 
                  className={`absolute left-0 top-0 h-full ${
                    rainData.timeToRain === 0 ? 'bg-blue-500' :
                    rainData.timeToRain < 30 ? 'bg-yellow-500' : 'bg-orange-500'
                  } transition-all duration-1000`}
                  style={{ 
                    width: `${Math.max(10, 100 - (rainData.timeToRain / 120 * 100))}%` 
                  }}
                ></div>
              </div>
              
              <div className="flex flex-col items-center">
                <Cloud className="w-8 h-8 text-blue-400 mb-2" />
                <span className="text-sm text-gray-400">Rain Cell</span>
              </div>
            </>
          )}
        </div>

        {rainData.timeToRain !== 999 && rainData.timeToRain !== '--' && (
          <div className="mt-4 text-center text-sm text-gray-400">
            {rainData.distance !== '--' && (
              <span>Rain cell is {rainData.distance}km away, moving {getDirectionName(rainData.direction)} at {rainData.speed}km/h</span>
            )}
          </div>
        )}
      </div>

      {/* Info Footer */}
      <div className="mt-4 text-center text-xs text-gray-500">
        Updates every 3 minutes • Based on radar analysis
      </div>
    </div>
  );
};

export default RainRadarCard;