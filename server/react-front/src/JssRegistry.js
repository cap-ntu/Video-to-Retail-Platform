import React from 'react';
import * as PropTypes from 'prop-types';
import JssProvider from 'react-jss/lib/JssProvider';
import { create } from 'jss';
import jssNested from 'jss-nested';
import {
  createMuiTheme,
  createGenerateClassName,
  jssPreset,
  MuiThemeProvider,
} from '@material-ui/core/styles';

const generateClassName = createGenerateClassName();
const jss = create({
  plugins: [jssPreset().plugins, jssNested()],
  ...jssPreset(),
});
jss.options.createGenerateClassName = createGenerateClassName;

const hysiaTheme = createMuiTheme({
  mixins: {
    toolbar: {
      minHeight: 48,
      '@media (min-width:0px) and (orientation: landscape)': {
        minHeight: 36,
      },
      '@media (min-width:600px)': {
        minHeight: 56,
      },
    },
  },
  palette: {
    primary: { main: '#33ACFF', contrastText: '#f9f9f9' },
    secondary: { main: '#66EFEF', contrastText: '#363D4E' },
    error: { main: '#F4606C' },
  },
  shape: {
    borderRadius: 8,
    buttonBorderRadius: 4,
    avatarBorderRadius: 16,
  },
  typography: {
    useNextVariants: true,
  },
});

const JssRegistry = ({ children }) => {
  return (
    <JssProvider jss={jss} generateClassName={generateClassName}>
      <MuiThemeProvider theme={hysiaTheme}>{children}</MuiThemeProvider>
    </JssProvider>
  );
};

JssRegistry.propTypes = {
  children: PropTypes.node.isRequired,
};

export default JssRegistry;
