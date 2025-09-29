/** @type {import('jest').Config} */
module.exports = {
	testEnvironment: 'jsdom',
	roots: ['<rootDir>'],
	testMatch: ['**/__tests__/**/*.test.js'],
	setupFiles: [],
	transform: {
		'^.+\\.js$': 'babel-jest',
	},
	transformIgnorePatterns: [
		'node_modules/(?!((@babel/runtime|@jest/transform|@jest/environment|@sinonjs/fake-timers)/.*))'
	],
	moduleFileExtensions: ['js'],
};

