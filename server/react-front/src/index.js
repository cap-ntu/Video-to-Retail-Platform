import React from 'react';
import * as ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import './asset/css/styles.css';
import 'filepond/dist/filepond.min.css';
import 'filepond-plugin-image-preview/dist/filepond-plugin-image-preview.css';
import 'dplayer/dist/DPlayer.min.css';
// file pond
import { registerPlugin } from 'filepond';
import FilePondPluginImagePreview from 'filepond-plugin-image-preview';
import FilePondPluginFileRename from 'filepond-plugin-file-rename';
import FilePondPluginFileValidateType from 'filepond-plugin-file-validate-type';
import registerServiceWorker from './registerServiceWorker';
import RootRouter from './urls';
import store from './redux/store/configStore';
import JssRegistry from './JssRegistry';

registerPlugin(
  FilePondPluginImagePreview,
  FilePondPluginFileRename,
  FilePondPluginFileValidateType,
);

ReactDOM.render(
  <JssRegistry>
    <Provider store={store}>
      <RootRouter />
    </Provider>
  </JssRegistry>,
  document.getElementById('root'),
);

registerServiceWorker();
